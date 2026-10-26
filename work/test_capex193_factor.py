"""Generated records only; never loads the admitted financial dataset."""
from copy import deepcopy
from datetime import date,timedelta
from decimal import localcontext
from fractions import Fraction
import hashlib,json,unittest
import numpy as np
import capex193_factor as f


def period(start,end):
    return dict(period_start=start,period_end=end,days_inclusive=(date.fromisoformat(end)-date.fromisoformat(start)).days+1)

def event(identifier='synthetic',symbol='AAPL'):
    periods=dict(current=period('2025-01-01','2025-12-31'),comparison=period('2024-01-01','2024-12-31'))
    source=dict(path='synthetic/report.html',sha256='a'*64,url='https://example.invalid/report')
    e=dict(event_id=identifier,symbol=symbol,fiscal_year=2025,release_date='2026-01-30',available_from='2026-02-02',
           identity=dict(issuer_id='synthetic_'+symbol,security_id=symbol,identity_valid_from='2026-02-02',
                         identity_valid_through=str(date(2025,12,31)+timedelta(days=550))),
           source=source,periods=periods,inherited_rejection_codes=[],fact_admitted=False,
           cash_quality_admission=dict(admitted=True,reasons=[],definition=f.old.DEFINITION),
           cash_ppe_admission=dict(admitted=True,reasons=[],definition=f.DEFINITION))
    e['fields']={side:{name:dict(value_decimal=value,scale='1',currency_literal='$',currency_iso=None,
                   source=deepcopy(source),period=deepcopy(periods[side]),available_from=e['available_from'],fact_admitted=False)
              for name,value in [('operating_cash_flow','30'),('net_income','10'),('total_assets','150' if side=='current' else '50')]}
              for side in ('current','comparison')}
    row='Purchases of property, plant and equipment (12)'
    cell=dict(signed_literal='(12)',signed_value_decimal='-12',positive_outflow_decimal='12',scale='1',
              currency_literal='$',currency_iso=None,row_label='Purchases of property, plant and equipment',
              row_text=row,row_text_sha256=hashlib.sha256(row.encode()).hexdigest(),
              locator=dict(format='html',period_end='2025-12-31'))
    e['cash_ppe_source_evidence']={k:deepcopy(e[k]) for k in ('event_id','symbol','release_date','available_from','source')}
    e['cash_ppe_source_evidence'].update(inherited_admitted=True,extraction_verified=True,status='verified',hold_reasons=[],capex=deepcopy(cell))
    e['cash_ppe']=dict(deepcopy(cell),source=deepcopy(source),period=deepcopy(periods['current']),available_from=e['available_from'],fact_admitted=False)
    return e

def sync(e):
    """Keep independently admitted synthetic producer facts consistent after changing inputs."""
    p=e['cash_ppe_source_evidence']
    for key in ('event_id','symbol','release_date','available_from','source'):p[key]=deepcopy(e[key])
    raw=p.get('field',p.get('capex'))
    for key in list(raw):raw[key]=deepcopy(e['cash_ppe'][key])
    return e

def fixture(dates=None):
    dates=dates if dates is not None else ['2026-02-02','2026-03-31','2026-04-30']
    names=['AAPL','AMD','MSFT','NVDA']+['S'+str(i) for i in range(28)]+['ETF'+str(i) for i in range(17)]
    return dict(dates=dates,symbols=[names[:] for _ in dates],etfs=names[32:]),dict(schema_version=1,method=f.METHOD,conditional_research_only=True,events=[event()])

def later_event():
    e=event('later');e['available_from']=e['identity']['identity_valid_from']='2026-03-15'
    for side in f.old.SIDES:
        for c in e['fields'][side].values():c['available_from']=e['available_from']
    e['cash_ppe']['available_from']=e['available_from'];return sync(e)

class Tests(unittest.TestCase):
    def evaluate(self,e=None,signal='2026-03-31'):
        return f.evaluate_report(event() if e is None else e,signal)
    def test_two_exact_definitions(self):
        r=self.evaluate();self.assertTrue(r['available'])
        self.assertEqual(r['control_bounded'],float(Fraction(3,13)))
        self.assertEqual(r['candidate_bounded'],float(Fraction(9,59)))
        self.assertNotEqual(r['candidate_bounded'],r['control_bounded']-float(Fraction(3,28)))
    def test_strict_old_and_new_gate(self):
        for name in ('cash_quality_admission','cash_ppe_admission'):
            for admitted in (False,None,1,'true'):
                e=event();e[name]['admitted']=admitted;self.assertFalse(self.evaluate(e)['available'])
            for change in ({'extra':True},{'definition':'wrong'},{'reasons':['hold']},{'reasons':{}}):
                e=event();e[name].update(change);self.assertFalse(self.evaluate(e)['available'])
            e=event();del e[name];self.assertFalse(self.evaluate(e)['available'])
    def test_source_and_clock_exact(self):
        for field,value in [('source',{'path':'different','sha256':'b'*64}),('period',period('2024-01-01','2024-12-31')),
                            ('available_from','2026-02-03'),('fact_admitted',True)]:
            e=event();e['cash_ppe'][field]=value;self.assertFalse(self.evaluate(e)['available'],field)
        for key,value in [('event_id','wrong'),('symbol','NVDA'),('release_date','2026-01-29'),('available_from','2026-02-03')]:
            e=event();e['cash_ppe_source_evidence'][key]=value;self.assertFalse(self.evaluate(e)['available'],key)
    def test_normalized_cell_preserves_evidence(self):
        e=event();e['cash_ppe']['positive_outflow_decimal']='11';e['cash_ppe']['signed_value_decimal']='-11'
        self.assertFalse(self.evaluate(e)['available'])
        e=event();e['cash_ppe_source_evidence']['field']=deepcopy(e['cash_ppe_source_evidence']['capex'])
        self.assertFalse(self.evaluate(e)['available'])
        e=event();e['cash_ppe_source_evidence']['field']=e['cash_ppe_source_evidence'].pop('capex')
        self.assertTrue(self.evaluate(e)['available'])
    def test_held_or_missing_evidence_blocks_both(self):
        for key,value in [('status','held'),('extraction_verified',False),('inherited_admitted',False),('hold_reasons',['hold'])]:
            e=event();e['cash_ppe_source_evidence'][key]=value;self.assertFalse(self.evaluate(e)['available'])
        for key in ('cash_ppe','cash_ppe_source_evidence'):
            e=event();e[key]=None;self.assertFalse(self.evaluate(e)['available'])
    def test_old_ni_gate_retained_but_not_ratio_input(self):
        e=event();e['fields']['current']['net_income']['value_decimal']='999999';self.assertEqual(self.evaluate(e),self.evaluate())
        e['fields']['current']['net_income']=None;self.assertFalse(self.evaluate(e)['available'])
        e=event();e['inherited_rejection_codes']=['hold'];self.assertFalse(self.evaluate(e)['available'])
    def test_comparison_flows_unused(self):
        e=event();e['fields']['comparison']['operating_cash_flow']=None;e['fields']['comparison']['net_income']={'poison':'unused'}
        self.assertEqual(self.evaluate(e),self.evaluate())
    def test_signed_cash_and_ppe_exceeding_cash(self):
        for cash in ('-30','0','5','12','30'):
            e=event();e['fields']['current']['operating_cash_flow']['value_decimal']=cash;r=self.evaluate(e)
            self.assertTrue(r['available'])
            for key,x in [('candidate_bounded',(Fraction(cash)-12)/100),('control_bounded',Fraction(cash)/100)]:
                self.assertEqual(r[key],float(x/(1+abs(x))))
            self.assertLessEqual(r['candidate_bounded'],r['control_bounded'])
    def test_zero_is_not_missing(self):
        e=event();e['cash_ppe'].update(signed_value_decimal='0',positive_outflow_decimal='0',signed_literal='0');sync(e)
        r=self.evaluate(e);self.assertTrue(r['available']);self.assertEqual(r['candidate_bounded'],r['control_bounded'])
        e['cash_ppe']=None;r=self.evaluate(e);self.assertFalse(r['available']);self.assertIsNone(r['control_bounded'])
    def test_ppe_sign_units_and_literal_reject(self):
        bad=[('signed_value_decimal','12'),('positive_outflow_decimal','-12'),('scale','0'),('scale','2'),
             ('currency_literal','EUR'),('currency_iso','EUR'),('signed_literal',''),('row_text_sha256','bad')]
        for key,value in bad:
            e=event();e['cash_ppe'][key]=value;sync(e);self.assertFalse(self.evaluate(e)['available'],(key,value))
        for value in (True,12,'NaN','Infinity','1,200','1e2'):
            e=event();e['cash_ppe']['positive_outflow_decimal']=value;sync(e);self.assertFalse(self.evaluate(e)['available'])
    def test_mixed_scope_and_wrong_locator_reject(self):
        for label in ('Purchases of property and equipment and intangibles','Principal payments on property and equipment','Free cash flow'):
            e=event();e['cash_ppe']['row_label']=label;sync(e);self.assertFalse(self.evaluate(e)['available'])
        e=event();e['cash_ppe']['locator']['period_end']='2024-12-31';sync(e);self.assertFalse(self.evaluate(e)['available'])
    def test_scaling_and_extreme_exact_cancellation(self):
        e=event()
        for side in f.old.SIDES:
            for c in e['fields'][side].values():c['scale']='1000000'
        e['cash_ppe']['scale']='1000000';sync(e);self.assertEqual(self.evaluate(e),self.evaluate())
        e=event();huge=10**100;e['fields']['current']['operating_cash_flow']['value_decimal']=str(huge+1)
        e['cash_ppe'].update(signed_value_decimal=str(-huge),positive_outflow_decimal=str(huge),signed_literal='('+str(huge)+')');sync(e)
        for side in f.old.SIDES:e['fields'][side]['total_assets']['value_decimal']=str(huge)
        with localcontext() as ctx:
            ctx.prec=2;r=self.evaluate(e)
        self.assertEqual(r['candidate_bounded'],float(Fraction(1,huge+1)))
        self.assertGreater(r['candidate_bounded'],0)
    def test_acquisition_of_ppe_is_not_business_acquisition(self):
        e=event();e['cash_ppe']['row_label']='Payments for acquisition of property, plant and equipment';sync(e)
        self.assertTrue(self.evaluate(e)['available'])
        for label in ('Purchases of property and equipment and acquisition of businesses',
                      'Payments for acquisition of property, plant and equipment and businesses',
                      'Payments for acquisition of property, plant and equipment and business acquisitions'):
            e['cash_ppe']['row_label']=label;sync(e);self.assertFalse(self.evaluate(e)['available'])
    def test_statement_currency_evidence_preserves_absent_row_literal(self):
        e=event();e['cash_ppe'].update(currency_literal=None,statement_currency_literal='$',
                    currency_scope='same formal cash-flow table; ISO unverified')
        loc=dict(format='html',table_index_zero_based=2,statement_segment='cashflow')
        e['fields']['current']['operating_cash_flow']['locator']=deepcopy(loc)
        e['cash_ppe']['locator'].update(loc,original_cfo_locator=deepcopy(loc));sync(e)
        self.assertEqual(self.evaluate(e),self.evaluate())
        self.assertIsNone(e['cash_ppe']['currency_literal'])
        for key,value in [('statement_currency_literal',None),('currency_scope','unknown'),('currency_literal','EUR')]:
            bad=deepcopy(e);bad['cash_ppe'][key]=value;sync(bad);self.assertFalse(self.evaluate(bad)['available'])
        e=event();e['cash_ppe']['statement_currency_literal']='EUR';sync(e)
        self.assertFalse(self.evaluate(e)['available'])
    def test_currency_structure_cannot_be_asserted_by_scope_text(self):
        for fmt,scope,key,value in [('html','table','table_index_zero_based',2),('pdf','page','page_1based',10)]:
            e=event();loc=dict(format=fmt,**{key:value})
            if fmt=='html':loc['statement_segment']='cashflow'
            e['fields']['current']['operating_cash_flow']['locator']=deepcopy(loc)
            e['cash_ppe'].update(currency_literal=None,statement_currency_literal='$',
                                currency_scope='same formal cash-flow '+scope+'; ISO unverified')
            e['cash_ppe']['locator'].update(loc,original_cfo_locator=deepcopy(loc));sync(e)
            self.assertTrue(self.evaluate(e)['available'])
            for mutation in ('index','original','format'):
                bad=deepcopy(e)
                if mutation=='index':bad['cash_ppe']['locator'][key]=value+1
                elif mutation=='original':bad['cash_ppe']['locator']['original_cfo_locator']={}
                else:bad['cash_ppe']['locator']['format']='unknown'
                sync(bad);self.assertFalse(self.evaluate(bad)['available'])
    def test_annual_identity_and_expiry(self):
        end=date(2025,12,31);self.assertTrue(self.evaluate(signal=str(end+timedelta(days=550)))['available'])
        self.assertFalse(self.evaluate(signal=str(end+timedelta(days=551)))['available'])
        for key,value in [('security_id','OTHER'),('identity_valid_from','2026-02-03'),('identity_valid_through','2030-01-01')]:
            e=event();e['identity'][key]=value;self.assertFalse(self.evaluate(e)['available'])
        e=event();e['periods']['current']['days_inclusive']=True;self.assertFalse(self.evaluate(e)['available'])
        for side in f.old.SIDES:
            e=event();e['fields'][side]['total_assets']['value_decimal']='0';self.assertFalse(self.evaluate(e)['available'])
    def test_release_and_availability(self):
        self.assertFalse(self.evaluate(signal='2026-01-30')['available'])
        self.assertTrue(self.evaluate(signal='2026-02-02')['available'])
        e=event();e['release_date']=e['available_from'];sync(e);self.assertFalse(self.evaluate(e)['available'])
    def test_axes_masks_input_immutability(self):
        meta,data=fixture();data['events'].append(event('amd','AMD'));saved=deepcopy((meta,data));r=f.build_features(meta,data)
        self.assertEqual((meta,data),saved);self.assertEqual(r['audit']['common_available_cells'],6)
        for key in f.EXTRA[:-1]:np.testing.assert_array_equal(np.isfinite(r['arrays'][key]),r['arrays']['common_available'])
        self.assertEqual(len(r['audit']['cells']),147);json.dumps(r['audit'],allow_nan=False)
        self.assertFalse(r['audit']['net_income_used_in_new_ratios']);self.assertEqual(r['audit']['new_model_fits'],0)
    def test_latest_held_or_missing_blocks_older(self):
        for change in ('hold','missing'):
            meta,data=fixture();new=later_event()
            if change=='hold':new['cash_ppe_admission']['admitted']=False;new['cash_ppe_admission']['reasons']=['new_hold']
            else:new['cash_ppe']=None
            data['events'].append(new);r=f.build_features(meta,data)
            self.assertTrue(r['arrays']['common_available'][0,0]);self.assertFalse(r['arrays']['common_available'][1:,0].any())
            self.assertEqual(r['audit']['cells'][49]['current_event_id'],'later')
    def test_future_facts_not_read_and_prefixes(self):
        meta,data=fixture();full=f.build_features(meta,data)
        data['events'].append(dict(event_id='future',symbol='AAPL',available_from='2030-01-02',cash_ppe=object()))
        for n in range(4):
            r=f.build_features(dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n]),data)
            for key in f.EXTRA:np.testing.assert_array_equal(r['arrays'][key],full['arrays'][key][:n])
            self.assertEqual(r['audit']['cells'],full['audit']['cells'][:49*n])
    def test_duplicate_or_ambiguous_reports(self):
        meta,data=fixture();data['events'].append(deepcopy(data['events'][0]))
        self.assertTrue(f.build_features(meta,data)['arrays']['common_available'][:,0].all())
        data['events'][-1]['event_id']='different';self.assertFalse(f.build_features(meta,data)['arrays']['common_available'][:,0].any())
        data['events'][-1]['event_id']='synthetic';data['events'][-1]['cash_ppe']=None
        self.assertEqual(f.build_features(meta,data)['audit']['cells'][0]['reason'],'conflicting_event_id')
    def test_etf_and_empty_records(self):
        meta,data=fixture();data['events'].append(event('etf','ETF0'));r=f.build_features(meta,data)
        self.assertFalse(r['arrays']['common_available'][:,32:].any())
        data['events']=[];r=f.build_features(meta,data);self.assertFalse(r['arrays']['common_available'].any())
        for key in f.EXTRA[:-1]:self.assertTrue(np.isnan(r['arrays'][key]).all())
    def test_dataset_and_axis_errors(self):
        for change in ({'method':'wrong'},{'schema_version':True},{'conditional_research_only':False}):
            meta,data=fixture();data.update(change)
            with self.assertRaises(ValueError):f.build_features(meta,data)
        meta,data=fixture();meta['symbols'][0][1]='AAPL'
        with self.assertRaises(ValueError):f.build_features(meta,data)
        meta,data=fixture();meta['dates']=list(reversed(meta['dates']))
        with self.assertRaises(ValueError):f.build_features(meta,data)

if __name__=='__main__':unittest.main()
