"""Synthetic statements only: independent source bindings, arithmetic and timing."""
from copy import deepcopy
from datetime import date,timedelta
from decimal import localcontext
from fractions import Fraction
import hashlib
import json
import unittest
import numpy as np
import monthly_trade_wc194_independent as o


def report(symbol='MSFT',identifier='oracle',style='explicit_html'):
    periods={side:dict(period_start=f'{year}-01-01',period_end=f'{year}-12-31',
                      days_inclusive=366 if year==2024 else 365)
             for side,year in [('current',2025),('comparison',2024)]}
    source={'path':'synthetic/current-annual.'+('pdf' if style=='pdf' else 'html'),'sha256':'b'*64}
    e=dict(event_id=identifier,symbol=symbol,release_date='2026-01-29',available_from='2026-01-30',
           identity=dict(issuer_id='company-'+symbol,security_id=symbol,identity_valid_from='2026-01-30',
                         identity_valid_through=str(date(2025,12,31)+timedelta(days=550))),
           periods=periods,source=source,inherited_rejection_codes=[],
           cash_quality_admission=dict(admitted=True,definition=o.OLD_DEFINITION,reasons=[]),
           trade_working_capital_admission=dict(admitted=True,definition=o.DEFINITION,reasons=[]))
    e['fields']={side:{field:dict(value_decimal=value,scale='1000',currency_literal='$',source=deepcopy(source),
                         period=deepcopy(periods[side]),available_from=e['available_from'])
                      for field,value in [('operating_cash_flow','42.5'),('net_income','-3'),
                                          ('total_assets','240' if side=='current' else '160')]}
                 for side in ('current','comparison')}
    cfo_loc=(dict(format='pdf',page_1based=3,line_1based=50) if style=='pdf' else
             dict(format='html',table_index_zero_based=3,statement_segment={'start_row':1,'end_row_exclusive':60},row_index_zero_based=50))
    e['fields']['current']['operating_cash_flow']['locator']=deepcopy(cfo_loc)
    e['trade_working_capital']={}
    for i,(key,label,value) in enumerate(zip(o.COMPONENTS,
            ['Accounts receivable, net','Inventories','Accounts payable'],['-4','-2','10'])):
        loc=dict(cfo_loc,original_cfo_locator=deepcopy(cfo_loc),period_end='2025-12-31')
        statement={k:deepcopy(v) for k,v in cfo_loc.items() if k not in ('row_index_zero_based','line_1based')}
        statement['original_cfo_locator']=deepcopy(cfo_loc)
        if style=='pdf':loc.update(line_1based=31+i,operating_changes_heading_line_1based=30)
        else:loc.update(row_index_zero_based=31+i,operating_changes_heading_row_zero_based=30)
        field=dict(component=key,signed_literal='('+value[1:]+')' if value.startswith('-') else value,
                   signed_value_decimal=value,scale='1000',row_label=label,
                   source=deepcopy(source),period=deepcopy(periods['current']),available_from=e['available_from'],
                   fact_admitted=False,currency_literal=None,currency_iso=None,
                   locator=loc,statement_currency_evidence=statement)
        field['row_text']=label+' '+field['signed_literal']
        field['row_text_sha256']=hashlib.sha256(field['row_text'].encode()).hexdigest()
        if style=='rows_html':
            statement.update(literal='$',rows=[dict(row_index_zero_based=5,row_text='Net income $42',row_html_sha256='c'*64)])
        else:
            field.update(statement_currency_literal='$',currency_scope='same formal cash-flow '+('page' if style=='pdf' else 'table')+'; ISO unverified')
        if style=='pdf':field.update(pdfium_text_agrees=True,pdfium_geometry={'synthetic':True})
        e['trade_working_capital'][key]=field
    e['trade_working_capital_source_evidence']={k:deepcopy(e[k]) for k in ('event_id','symbol','source','release_date','available_from')}
    e['trade_working_capital_source_evidence'].update(inherited_admitted=True,extraction_verified=True,status='verified',hold_reasons=[],fields=deepcopy(e['trade_working_capital']))
    return e


def sync(e):
    evidence=e['trade_working_capital_source_evidence']
    for key in ('event_id','symbol','source','release_date','available_from'):evidence[key]=deepcopy(e[key])
    evidence['fields']=deepcopy(e['trade_working_capital'])
    return e


def set_values(e,values):
    for key,value in zip(o.COMPONENTS,values):
        value=str(value);cell=e['trade_working_capital'][key]
        cell['signed_value_decimal']=value
        cell['signed_literal']='('+value[1:]+')' if value.startswith('-') else value
        cell['row_text']=cell['row_label']+' '+cell['signed_literal']
        cell['row_text_sha256']=hashlib.sha256(cell['row_text'].encode()).hexdigest()
    return sync(e)


def fixture(style='explicit_html'):
    names=['MSFT','AAPL','AMD','NVDA']+['STOCK'+str(i) for i in range(28)]+['ETF'+str(i) for i in range(17)]
    dates=['2026-01-30','2026-03-31','2026-05-29']
    meta=dict(dates=dates,symbols=[names[:] for _ in dates],etfs=names[32:])
    data=dict(schema_version=1,method=o.METHOD,conditional_research_only=True,events=[report(style=style)])
    return meta,data


class Tests(unittest.TestCase):
    def evaluate(self,e=None,signal='2026-03-31'):
        return o.evaluate_oracle(report() if e is None else e,signal)

    def test_independent_closed_form_decimal_context(self):
        with localcontext() as context:
            context.prec=2;r=self.evaluate()
        self.assertTrue(r['available']);self.assertEqual(r['candidate_bounded'],float(Fraction(77,477)))
        self.assertEqual(r['control_bounded'],float(Fraction(17,97)))

    def test_signed_components_not_absolute_or_reversed(self):
        for values in [(-30,2,3),(20,3,4),(0,0,0),(-4,-2,6)]:
            r=self.evaluate(set_values(report(),values));n=Fraction('42.5')-sum(map(Fraction,values))
            self.assertTrue(r['available']);self.assertEqual(r['candidate_bounded'],float(n/(200+abs(n))))
            self.assertEqual(r['control_bounded'],float(Fraction(17,97)))
        self.assertGreater(self.evaluate(set_values(report(),(-30,2,3)))['candidate_bounded'],self.evaluate()['control_bounded'])

    def test_sum_before_bounded_mapping_and_exact_cancellation(self):
        e=set_values(report(),(10**100,-10**100,1));e['fields']['current']['operating_cash_flow']['value_decimal']='2'
        self.assertEqual(self.evaluate(e)['candidate_bounded'],float(Fraction(1,201)))
        e=set_values(report(),(1,-1,0));r=self.evaluate(e);self.assertEqual(r['candidate_bounded'],r['control_bounded'])

    def test_currency_scale_invariance(self):
        e=report()
        for side in e['fields'].values():
            for cell in side.values():cell['scale']='1'
        for cell in e['trade_working_capital'].values():cell['scale']='1'
        self.assertEqual(self.evaluate(sync(e)),self.evaluate())

    def test_zero_dash_distinct_from_missing_component(self):
        e=set_values(report(),(0,0,0))
        for cell in e['trade_working_capital'].values():cell['signed_literal']='—'
        r=self.evaluate(sync(e));self.assertTrue(r['available']);self.assertEqual(r['candidate_bounded'],r['control_bounded'])
        for key in o.COMPONENTS:
            e=report();e['trade_working_capital'][key]=None;self.assertFalse(self.evaluate(e)['available'])

    def test_all_three_plain_values_and_literal_signs(self):
        for literal,decimal in [('(12)','-12'),('$ 12','12'),('/dollarsign (12)','-12'),('−12','-12'),('1,234','1234')]:
            e=report();e['trade_working_capital']['inventories'].update(signed_literal=literal,signed_value_decimal=decimal)
            self.assertTrue(self.evaluate(sync(e))['available'])
        for literal,decimal in [('(12)','12'),('1 2','12'),('(-12)','-12'),('1,23','123'),('NaN','0'),('12)','12')]:
            e=report();e['trade_working_capital']['inventories'].update(signed_literal=literal,signed_value_decimal=decimal)
            self.assertFalse(self.evaluate(sync(e))['available'])

    def test_both_admissions_and_held_source(self):
        for gate in ('cash_quality_admission','trade_working_capital_admission'):
            for value in ({'admitted':1},{'admitted':False},{'reasons':['hold']},{'definition':'wrong'},{'extra':True}):
                e=report();e[gate].update(value);self.assertFalse(self.evaluate(e)['available'])
        for key,value in [('inherited_admitted',False),('extraction_verified',False),('status','held'),('hold_reasons',['hold'])]:
            e=report();e['trade_working_capital_source_evidence'][key]=value;self.assertFalse(self.evaluate(e)['available'])

    def test_latest_held_blocks_older(self):
        meta,data=fixture();new=report(identifier='latest');new['available_from']='2026-03-01'
        new['trade_working_capital_admission']['admitted']=False;data['events'].append(new)
        built=o.independent_features(meta,data);self.assertTrue(built['arrays']['common_available'][0,0])
        self.assertFalse(built['arrays']['common_available'][1:,0].any())
        self.assertEqual(built['audit']['cells'][49]['current_event_id'],'latest')

    def test_ni_gate_but_no_comparison_flows(self):
        e=report();e['fields']['current']['net_income']['value_decimal']='1e300';self.assertEqual(self.evaluate(e),self.evaluate())
        e['fields']['current']['net_income']=None;self.assertEqual(self.evaluate(e)['reason'],'missing_current_net_income')
        e=report();e['fields']['comparison']['net_income']=object();e['fields']['comparison']['operating_cash_flow']=object()
        self.assertEqual(self.evaluate(e),self.evaluate())

    def test_assets_positive_same_report(self):
        for side in ('current','comparison'):
            for value in ('0','-1'):
                e=report();e['fields'][side]['total_assets']['value_decimal']=value;self.assertFalse(self.evaluate(e)['available'])
        e=report();e['fields']['comparison']['total_assets']['source']={};self.assertFalse(self.evaluate(e)['available'])

    def test_annual_period_and_expiry_inclusive(self):
        end=date(2025,12,31)
        self.assertTrue(self.evaluate(signal=str(end+timedelta(days=550)))['available'])
        self.assertFalse(self.evaluate(signal=str(end+timedelta(days=551)))['available'])
        for key,value in [('security_id','AMD'),('identity_valid_from','2026-01-29'),('identity_valid_through','2030-01-01')]:
            e=report();e['identity'][key]=value;self.assertFalse(self.evaluate(e)['available'])
        e=report();e['periods']['current']['days_inclusive']=True;self.assertFalse(self.evaluate(e)['available'])
        e=report();e['periods']['comparison']['period_end']='2024-12-30';self.assertFalse(self.evaluate(e)['available'])

    def test_normalized_source_clock_and_raw_evidence(self):
        for key,value in [('source',{}),('period',{}),('available_from','2026-02-01'),('fact_admitted',True)]:
            e=report();e['trade_working_capital']['inventories'][key]=value;sync(e);self.assertFalse(self.evaluate(e)['available'])
        e=report();e['trade_working_capital']['inventories']['signed_value_decimal']='123';self.assertFalse(self.evaluate(e)['available'])
        e=report();e['trade_working_capital_source_evidence']['event_id']='different';self.assertFalse(self.evaluate(e)['available'])

    def test_mixed_scope_and_duplicate_rows_rejected(self):
        badlabels={'accounts_receivable':'Vendor non-trade receivables','inventories':'Inventories and other assets',
                   'accounts_payable':'Accounts payable and accrued liabilities'}
        for key,label in badlabels.items():
            e=report();e['trade_working_capital'][key]['row_label']=label;self.assertFalse(self.evaluate(sync(e))['available'])
        e=report();e['trade_working_capital']['inventories']['locator']['row_index_zero_based']=31
        self.assertFalse(self.evaluate(sync(e))['available'])

    def test_statement_currency_three_native_paths(self):
        for style in ('explicit_html','rows_html','pdf'):
            self.assertEqual(self.evaluate(report(style=style)),self.evaluate())
        for key,value in [('currency_iso','USD'),('currency_literal','EUR'),('statement_currency_literal','EUR'),('currency_scope','BS')]:
            e=report();e['trade_working_capital']['inventories'][key]=value
            self.assertFalse(self.evaluate(sync(e))['available'])

    def test_same_statement_locator_conflicts(self):
        for key,value in [('table_index_zero_based',4),('table_index_zero_based',True),('statement_segment','other'),('original_cfo_locator',{})]:
            e=report();e['trade_working_capital']['inventories']['locator'][key]=value
            self.assertFalse(self.evaluate(sync(e))['available'])
        e=report();e['trade_working_capital']['inventories']['statement_currency_evidence']['statement_segment']='other'
        self.assertFalse(self.evaluate(sync(e))['available'])
        e=report(style='pdf');e['trade_working_capital']['inventories']['locator']['page_1based']=4
        self.assertFalse(self.evaluate(sync(e))['available'])
        e=report(style='rows_html');e['trade_working_capital']['inventories']['statement_currency_evidence']['rows'][0]['row_index_zero_based']=True
        self.assertFalse(self.evaluate(sync(e))['available'])

    def test_nonfinite_numeric_scale_hash(self):
        for key,value in [('scale','0'),('scale','1'),('signed_value_decimal','NaN'),('signed_value_decimal','1e6'),('row_text_sha256','0'*64)]:
            e=report();e['trade_working_capital']['inventories'][key]=value
            self.assertFalse(self.evaluate(sync(e))['available'])

    def test_future_poison_all_prefixes(self):
        meta,data=fixture();full=o.independent_features(meta,data)
        data['events'].append(dict(event_id='future',symbol='MSFT',available_from='2030-01-02',trade_working_capital=object()))
        for n in range(4):
            short=dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n]);got=o.independent_features(short,data)
            for key in o.KEYS:np.testing.assert_array_equal(got['arrays'][key],full['arrays'][key][:n])
            self.assertEqual(got['audit']['cells'],full['audit']['cells'][:n*49])

    def test_conflicting_and_ambiguous_reports_etf_neutral(self):
        meta,data=fixture();data['events'].append(deepcopy(data['events'][0]))
        self.assertTrue(o.independent_features(meta,data)['arrays']['common_available'][:,0].all())
        data['events'][-1]['event_id']='different';self.assertFalse(o.independent_features(meta,data)['arrays']['common_available'].any())
        data['events']=[report('ETF0')];self.assertFalse(o.independent_features(meta,data)['arrays']['common_available'].any())

    def test_verifier_corrupt_values_and_audit(self):
        meta,data=fixture();built=o.independent_features(meta,data);r=o.verify_features(meta,data,built)
        self.assertEqual((r['checked_cells'],r['available_cells'],r['symbols_per_event']),(147,3,49))
        self.assertTrue(r['trade_working_capital_scope_checked'])
        for key in o.KEYS:
            bad=deepcopy(built);bad['arrays'][key][0,0]=False if key=='common_available' else .5
            with self.assertRaises(ValueError):o.verify_features(meta,data,bad)
        bad=deepcopy(built);bad['audit']['cells'][0]['current_days']=1
        with self.assertRaises(ValueError):o.verify_features(meta,data,bad)

    def test_eligible_only_deltas_no_initial_suppression(self):
        meta,data=fixture();a=o.independent_features(meta,data)['arrays'];eligibility=np.ones((3,49),bool);eligibility[1,0]=False
        d=o.independent_deltas(a,eligibility);r=o.verify_deltas(a,d,eligibility)
        self.assertTrue(r['eligible_only']);self.assertEqual(r['changed_cells'][o.ARMS[2]],2)
        self.assertNotEqual(d[o.ARMS[2]][0,0],0);self.assertEqual(d[o.ARMS[2]][1,0],0)
        d[o.ARMS[2]][0,0]=0
        with self.assertRaises(ValueError):o.verify_deltas(a,d,eligibility)
        with self.assertRaises(ValueError):o.independent_deltas(a,eligibility.astype(int))
        with self.assertRaises(ValueError):o.independent_deltas(a,eligibility,.02)

    def test_schema_immutability_and_receipt_serializable(self):
        meta,data=fixture();snapshot=deepcopy((meta,data));built=o.independent_features(meta,data)
        self.assertEqual((meta,data),snapshot);json.dumps(built['audit'],allow_nan=False)
        data['schema_version']=True
        with self.assertRaises(ValueError):o.independent_features(meta,data)
        meta,data=fixture();meta['symbols'][0][1]='MSFT'
        with self.assertRaises(ValueError):o.independent_features(meta,data)

    def test_pinned_accounting_and_no_producer_import(self):
        self.assertTrue(callable(o.accounting().audit_path));self.assertNotIn('monthly_trade_wc194_factor',o.__dict__)
        self.assertEqual(o.frozen_helper().FROZEN_HELPER_SHA,'e83305d9f784ef9aa5ae0554cd1deb558d927b4182010af599b977efa69618dc')

    def test_producer_cross_both_synthetic_generators(self):
        import monthly_trade_wc194_factor as producer
        import test_monthly_trade_wc194_factor as pt
        def producer_fixture(style='explicit_html'):
            meta,data=pt.fixture();data['events']=[pt.event(style=style)];return meta,data
        for make in (fixture,producer_fixture):
            for style in ('explicit_html','rows_html','pdf'):
                meta,data=make(style=style)
                for n in range(len(meta['dates'])+1):
                    short=dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n])
                    o.verify_features(short,data,producer.build_features(short,data))
            for mutation in ('missing','held','currency','scope','identity'):
                meta,data=make();e=data['events'][0]
                if mutation=='missing':e['trade_working_capital']['inventories']=None
                elif mutation=='held':e['cash_quality_admission']['admitted']=False
                elif mutation=='currency':e['trade_working_capital']['inventories']['currency_iso']='USD'
                elif mutation=='scope':e['trade_working_capital']['inventories']['row_label']='Inventories and other assets'
                else:e['identity']['security_id']='WRONG'
                o.verify_features(meta,data,producer.build_features(meta,data))


if __name__=='__main__':unittest.main()
