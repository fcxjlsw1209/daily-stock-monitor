"""Generated records only. Does not load any admitted financial data."""
from copy import deepcopy
from datetime import date,timedelta
from decimal import localcontext
from fractions import Fraction
import hashlib,json,unittest
import numpy as np
import monthly_trade_wc194_factor as f


def period(start,end):
 return dict(period_start=start,period_end=end,days_inclusive=(date.fromisoformat(end)-date.fromisoformat(start)).days+1)


def event(identifier='synthetic',symbol='AAPL',style='explicit_html'):
 periods=dict(current=period('2025-01-01','2025-12-31'),comparison=period('2024-01-01','2024-12-31'))
 source=dict(path='synthetic/report.html',sha256='a'*64,url='https://example.invalid/report')
 e=dict(event_id=identifier,symbol=symbol,fiscal_year=2025,release_date='2026-01-30',available_from='2026-02-02',
  identity=dict(issuer_id='synthetic_'+symbol,security_id=symbol,identity_valid_from='2026-02-02',
                identity_valid_through=str(date(2025,12,31)+timedelta(days=550))),source=source,periods=periods,
  inherited_rejection_codes=[],fact_admitted=False,
  cash_quality_admission=dict(admitted=True,reasons=[],definition=f.old.DEFINITION),
  trade_working_capital_admission=dict(admitted=True,reasons=[],definition=f.DEFINITION))
 e['fields']={side:{name:dict(value_decimal=value,scale='1',currency_literal='$',currency_iso=None,
  source=deepcopy(source),period=deepcopy(periods[side]),available_from=e['available_from'],fact_admitted=False)
  for name,value in [('operating_cash_flow','30'),('net_income','10'),('total_assets','150' if side=='current' else '50')]}
  for side in ('current','comparison')}
 if style=='pdf':cfo=dict(format='pdf',page_1based=3,line_1based=24,period_end='2025-12-31')
 else:cfo=dict(format='html',table_index_zero_based=2,row_index_zero_based=24,period_end='2025-12-31')
 e['fields']['current']['operating_cash_flow']['locator']=deepcopy(cfo)
 e['trade_working_capital']={}
 for i,(key,label,value) in enumerate(zip(f.COMPONENTS,('Accounts receivable, net','Inventories','Accounts payable'),(-20,-10,15))):
  literal='('+str(-value)+')' if value<0 else str(value);row=label+' '+literal
  locator=dict(deepcopy(cfo),original_cfo_locator=deepcopy(cfo))
  locator['line_1based' if style=='pdf' else 'row_index_zero_based']=10+i
  statement={k:deepcopy(v) for k,v in cfo.items() if k in ('format','page_1based','table_index_zero_based')}
  statement['original_cfo_locator']=deepcopy(cfo)
  cell=dict(component=key,signed_literal=literal,signed_value_decimal=str(value),scale='1',currency_literal=None,
   currency_iso=None,row_label=label,row_text=row,row_text_sha256=hashlib.sha256(row.encode()).hexdigest(),
   locator=locator,statement_currency_evidence=statement,source=deepcopy(source),period=deepcopy(periods['current']),
   available_from=e['available_from'],fact_admitted=False)
  if style=='rows_html':statement.update(literal='$',rows=[dict(row_index_zero_based=3,row_text='Net income $ 10',row_html_sha256='b'*64)])
  else:cell.update(statement_currency_literal='$',currency_scope='same formal cash-flow '+('page' if style=='pdf' else 'table')+'; ISO unverified')
  e['trade_working_capital'][key]=cell
 e['trade_working_capital_source_evidence']={k:deepcopy(e[k]) for k in ('event_id','symbol','release_date','available_from','source')}
 e['trade_working_capital_source_evidence'].update(inherited_admitted=True,extraction_verified=True,status='verified',hold_reasons=[],fields=deepcopy(e['trade_working_capital']))
 return e


def sync(e):
 p=e['trade_working_capital_source_evidence']
 for key in ('event_id','symbol','release_date','available_from','source'):p[key]=deepcopy(e[key])
 p['fields']=deepcopy(e['trade_working_capital'])
 return e


def set_values(e,values):
 for key,value in zip(f.COMPONENTS,values):
  cell=e['trade_working_capital'][key];text=str(value)
  cell['signed_value_decimal']=text;cell['signed_literal']='('+text[1:]+')' if text.startswith('-') else text
  cell['row_text']=cell['row_label']+' '+cell['signed_literal'];cell['row_text_sha256']=hashlib.sha256(cell['row_text'].encode()).hexdigest()
 return sync(e)


def fixture(dates=None):
 dates=dates if dates is not None else ['2026-02-02','2026-03-31','2026-04-30']
 names=['AAPL','AMD','MSFT','NVDA']+['S'+str(i) for i in range(28)]+['ETF'+str(i) for i in range(17)]
 return dict(dates=dates,symbols=[names[:] for _ in dates],etfs=names[32:]),dict(schema_version=1,method=f.METHOD,conditional_research_only=True,events=[event()])


def later_event():
 e=event('later');e['available_from']=e['identity']['identity_valid_from']='2026-03-15'
 for side in f.old.SIDES:
  for c in e['fields'][side].values():c['available_from']=e['available_from']
 for c in e['trade_working_capital'].values():c['available_from']=e['available_from']
 return sync(e)


class FormulaTests(unittest.TestCase):
 def evaluate(self,e=None,signal='2026-03-31'):return f.evaluate_report(event() if e is None else e,signal)
 def test_exact_signed_three_component_definition(self):
  r=self.evaluate();self.assertTrue(r['available']);self.assertEqual(r['control_bounded'],float(Fraction(3,13)))
  self.assertEqual(r['candidate_bounded'],float(Fraction(9,29)))
  self.assertNotEqual(r['candidate_bounded'],r['control_bounded']+float(Fraction(3,23)))
 def test_positive_negative_and_offsetting_contributions(self):
  for values in ((20,10,15),(-20,-10,-15),(-20,-10,15),(0,0,0),(20,-25,5)):
   e=set_values(event(),values);r=self.evaluate(e);self.assertTrue(r['available']);x=Fraction(30-sum(values),100)
   self.assertEqual(r['candidate_bounded'],float(x/(1+abs(x))))
   self.assertEqual(r['control_bounded'],float(Fraction(3,13)))
  self.assertEqual(self.evaluate(set_values(event(),(20,-25,5)))['candidate_bounded'],float(Fraction(3,13)))
 def test_signed_and_zero_cfo(self):
  for cash in ('-30','0','5','30'):
   e=event();e['fields']['current']['operating_cash_flow']['value_decimal']=cash;r=self.evaluate(e)
   self.assertTrue(r['available']);q=(Fraction(cash)+15)/100
   self.assertEqual(r['candidate_bounded'],float(q/(1+abs(q))))
 def test_exact_cancellation_no_decimal_context_rounding(self):
  huge=10**100;e=set_values(event(),(huge,-huge+1,0));e['fields']['current']['operating_cash_flow']['value_decimal']='2'
  for side in f.old.SIDES:e['fields'][side]['total_assets']['value_decimal']=str(huge)
  with localcontext() as ctx:
   ctx.prec=2;r=self.evaluate(e)
  self.assertEqual(r['candidate_bounded'],float(Fraction(1,huge+1)));self.assertGreater(r['candidate_bounded'],0)
 def test_scale_invariance(self):
  e=event()
  for side in f.old.SIDES:
   for c in e['fields'][side].values():c['scale']='1000000'
  for c in e['trade_working_capital'].values():c['scale']='1000000'
  sync(e);self.assertEqual(self.evaluate(e),self.evaluate())
 def test_dash_zero_is_not_missing(self):
  for key in f.COMPONENTS:
   e=set_values(event(),(0,0,0));e['trade_working_capital'][key]['signed_literal']='—';sync(e)
   r=self.evaluate(e);self.assertTrue(r['available']);self.assertEqual(r['candidate_bounded'],r['control_bounded'])
   e['trade_working_capital'][key]=None;sync(e);r=self.evaluate(e)
   self.assertFalse(r['available']);self.assertIsNone(r['control_bounded'])
 def test_all_components_required_same_support(self):
  for key in f.COMPONENTS:
   e=event();del e['trade_working_capital'][key];sync(e);self.assertFalse(self.evaluate(e)['available'])
  e=event();e['trade_working_capital']['other']=deepcopy(e['trade_working_capital']['inventories']);sync(e)
  self.assertFalse(self.evaluate(e)['available'])
 def test_ni_retained_not_used_in_ratio(self):
  e=event();e['fields']['current']['net_income']['value_decimal']='999999';self.assertEqual(self.evaluate(e),self.evaluate())
  e['fields']['current']['net_income']=None;self.assertFalse(self.evaluate(e)['available'])
  e=event();e['inherited_rejection_codes']=['old_hold'];self.assertFalse(self.evaluate(e)['available'])
 def test_comparison_flows_unused(self):
  e=event();e['fields']['comparison']['operating_cash_flow']=None;e['fields']['comparison']['net_income']=object()
  self.assertEqual(self.evaluate(e),self.evaluate())
 def test_no_duration_annualization(self):
  # Original guard permits53-week years but does not scale annual monetary flows.
  e=event();p=period('2025-01-01','2026-01-06');e['periods']['current']=p
  e['identity']['identity_valid_through']=str(date(2026,1,6)+timedelta(days=550))
  for c in e['fields']['current'].values():c['period']=deepcopy(p)
  cfo=e['fields']['current']['operating_cash_flow']['locator'];cfo['period_end']='2026-01-06'
  for c in e['trade_working_capital'].values():
   c['period']=deepcopy(p);c['locator']['period_end']='2026-01-06';c['locator']['original_cfo_locator']=deepcopy(cfo);c['statement_currency_evidence']['original_cfo_locator']=deepcopy(cfo)
  sync(e);r=self.evaluate(e);self.assertTrue(r['available']);self.assertEqual(r['candidate_bounded'],self.evaluate()['candidate_bounded'])


class EvidenceTests(unittest.TestCase):
 evaluate=FormulaTests.evaluate
 def test_gates(self):
  for name in ('cash_quality_admission','trade_working_capital_admission'):
   for admitted in (False,None,1,'true'):
    e=event();e[name]['admitted']=admitted;self.assertFalse(self.evaluate(e)['available'])
   for change in ({'extra':True},{'definition':'wrong'},{'reasons':['hold']},{'reasons':{}}):
    e=event();e[name].update(change);self.assertFalse(self.evaluate(e)['available'])
   e=event();del e[name];self.assertFalse(self.evaluate(e)['available'])
 def test_normalized_source_evidence_immutable(self):
  e=event();e['trade_working_capital']['inventories']['signed_value_decimal']='-11';self.assertFalse(self.evaluate(e)['available'])
  for key,value in [('status','held'),('inherited_admitted',False),('extraction_verified',False),('hold_reasons',['hold'])]:
   e=event();e['trade_working_capital_source_evidence'][key]=value;self.assertFalse(self.evaluate(e)['available'])
 def test_event_metadata_and_cell_binding(self):
  for key,value in [('event_id','wrong'),('symbol','NVDA'),('release_date','2026-01-29'),('available_from','2026-02-03'),('source',{})]:
   e=event();e['trade_working_capital_source_evidence'][key]=value;self.assertFalse(self.evaluate(e)['available'])
  for key,value in [('source',{}),('period',period('2024-01-01','2024-12-31')),('available_from','2026-02-03'),('fact_admitted',True)]:
   e=event();e['trade_working_capital']['accounts_payable'][key]=value;sync(e);self.assertFalse(self.evaluate(e)['available'])
 def test_signed_literal_parity_and_malformed_numbers(self):
  for value in (True,12,'NaN','Infinity','1,200','1e2',''):
   e=event();e['trade_working_capital']['inventories']['signed_value_decimal']=value;sync(e);self.assertFalse(self.evaluate(e)['available'])
  for value in ('10','(10','10)','(-10)','1 0','(—)',''):
   e=event();e['trade_working_capital']['inventories']['signed_literal']=value;sync(e);self.assertFalse(self.evaluate(e)['available'])
  e=event();e['trade_working_capital']['inventories']['signed_literal']='−10';sync(e);self.assertTrue(self.evaluate(e)['available'])
 def test_component_units_and_identity(self):
  for key,value in [('scale','0'),('scale','2'),('component','accounts_payable'),('row_text_sha256','bad')]:
   e=event();e['trade_working_capital']['inventories'][key]=value;sync(e);self.assertFalse(self.evaluate(e)['available'])
 def test_pure_trade_labels(self):
  for key,label in [('accounts_payable','Accounts payable, accrued liabilities and other'),('accounts_receivable','Vendor non-trade receivables'),
                    ('accounts_receivable','Accounts receivable and other assets'),('inventories','Amortization of inventory fair value adjustment')]:
   e=event();e['trade_working_capital'][key]['row_label']=label;sync(e);self.assertFalse(self.evaluate(e)['available'])
 def test_both_native_currency_evidence_paths(self):
  for style in ('explicit_html','rows_html','pdf'):
   self.assertEqual(self.evaluate(event(style=style)),self.evaluate())
   self.assertIsNone(event(style=style)['trade_working_capital']['accounts_receivable']['currency_literal'])
 def test_currency_conflicts_and_missing(self):
  for style in ('explicit_html','rows_html','pdf'):
   for key,value in [('currency_literal','EUR'),('currency_iso','EUR'),('currency_iso','USD'),('statement_currency_literal','EUR'),('currency_scope','unknown')]:
    e=event(style=style);e['trade_working_capital']['inventories'][key]=value;sync(e);self.assertFalse(self.evaluate(e)['available'])
  for mutation in ('rows','literal','hash','scope','statement'):
   e=event(style='rows_html');c=e['trade_working_capital']['inventories']
   if mutation=='rows':c['statement_currency_evidence']['rows']=[]
   elif mutation=='literal':c['statement_currency_evidence']['literal']='EUR'
   elif mutation=='hash':c['statement_currency_evidence']['rows'][0]['row_html_sha256']='bad'
   elif mutation=='scope':c['currency_scope']='same formal cash-flow page; ISO unverified'
   else:c['statement_currency_evidence']=None
   sync(e);self.assertFalse(self.evaluate(e)['available'])
 def test_same_statement_not_scope_assertion(self):
  for style in ('explicit_html','rows_html','pdf'):
   for mutation in ('location','original','format','period','statement_location'):
    e=event(style=style);c=e['trade_working_capital']['accounts_payable'];key='page_1based' if style=='pdf' else 'table_index_zero_based'
    if mutation=='location':c['locator'][key]+=1
    elif mutation=='original':c['locator']['original_cfo_locator']={}
    elif mutation=='format':c['locator']['format']='unknown'
    elif mutation=='period':c['locator']['period_end']='2024-12-31'
    else:c['statement_currency_evidence'][key]+=1
    sync(e);self.assertFalse(self.evaluate(e)['available'])
 def test_three_distinct_rows(self):
  e=event();e['trade_working_capital']['inventories']['locator']['row_index_zero_based']=10;sync(e)
  self.assertFalse(self.evaluate(e)['available'])
 def test_statement_segment_mismatch(self):
  e=event();e['trade_working_capital']['inventories']['locator']['statement_segment']={'start_row':100};sync(e)
  self.assertFalse(self.evaluate(e)['available'])


class ClockAndBuildTests(unittest.TestCase):
 def test_annual_identity_and_expiry(self):
  end=date(2025,12,31);self.assertTrue(f.evaluate_report(event(),str(end+timedelta(days=550)))['available'])
  self.assertFalse(f.evaluate_report(event(),str(end+timedelta(days=551)))['available'])
  for key,value in [('security_id','OTHER'),('identity_valid_from','2026-02-03'),('identity_valid_through','2030-01-01')]:
   e=event();e['identity'][key]=value;self.assertFalse(f.evaluate_report(e,'2026-03-31')['available'])
  e=event();e['periods']['current']['days_inclusive']=True;self.assertFalse(f.evaluate_report(e,'2026-03-31')['available'])
  for side in f.old.SIDES:
   e=event();e['fields'][side]['total_assets']['value_decimal']='0';self.assertFalse(f.evaluate_report(e,'2026-03-31')['available'])
 def test_release_and_availability(self):
  self.assertFalse(f.evaluate_report(event(),'2026-01-30')['available']);self.assertTrue(f.evaluate_report(event(),'2026-02-02')['available'])
  e=event();e['release_date']=e['available_from'];sync(e);self.assertFalse(f.evaluate_report(e,'2026-02-02')['available'])
 def test_full_axes_masks_immutability(self):
  meta,data=fixture();data['events'].append(event('amd','AMD',style='rows_html'));saved=deepcopy((meta,data));built=f.build_features(meta,data)
  self.assertEqual((meta,data),saved);self.assertEqual(built['audit']['common_available_cells'],6)
  for key in f.EXTRA[:-1]:np.testing.assert_array_equal(np.isfinite(built['arrays'][key]),built['arrays']['common_available'])
  self.assertTrue(built['arrays']['common_available'][0,0]);self.assertNotEqual(built['arrays']['candidate_bounded'][0,0],0)
  self.assertEqual(len(built['audit']['cells']),147);json.dumps(built['audit'],allow_nan=False)
 def test_latest_held_missing_or_expired_blocks_older(self):
  for mutation in ('held','missing','invalid'):
   meta,data=fixture();new=later_event()
   if mutation=='held':new['trade_working_capital_admission'].update(admitted=False,reasons=['hold'])
   elif mutation=='missing':new['trade_working_capital']['inventories']=None
   else:new['trade_working_capital']['inventories']['scale']='2'
   data['events'].append(new);built=f.build_features(meta,data)
   self.assertTrue(built['arrays']['common_available'][0,0]);self.assertFalse(built['arrays']['common_available'][1:,0].any())
   self.assertEqual(built['audit']['cells'][49]['current_event_id'],'later')
 def test_future_numeric_poison_and_all_prefixes(self):
  meta,data=fixture();full=f.build_features(meta,data)
  data['events'].append(dict(event_id='future',symbol='AAPL',available_from='2030-01-02',trade_working_capital=object()))
  for n in range(4):
   built=f.build_features(dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n]),data)
   for key in f.EXTRA:np.testing.assert_array_equal(built['arrays'][key],full['arrays'][key][:n])
   self.assertEqual(built['audit']['cells'],full['audit']['cells'][:49*n])
 def test_duplicate_and_ambiguous_latest(self):
  meta,data=fixture();data['events'].append(deepcopy(data['events'][0]));self.assertTrue(f.build_features(meta,data)['arrays']['common_available'][:,0].all())
  data['events'][-1]['event_id']='different';self.assertFalse(f.build_features(meta,data)['arrays']['common_available'][:,0].any())
  data['events'][-1]['event_id']='synthetic';data['events'][-1]['trade_working_capital']=None
  self.assertEqual(f.build_features(meta,data)['audit']['cells'][0]['reason'],'conflicting_event_id')
 def test_etfs_and_missing_neutral(self):
  meta,data=fixture();data['events'].append(event('etf','ETF0'));built=f.build_features(meta,data);self.assertFalse(built['arrays']['common_available'][:,32:].any())
  data['events']=[];built=f.build_features(meta,data)
  self.assertFalse(built['arrays']['common_available'].any())
  for key in f.EXTRA[:-1]:self.assertTrue(np.isnan(built['arrays'][key]).all())
 def test_dataset_and_axis_validation(self):
  for change in ({'schema_version':True},{'method':'wrong'},{'conditional_research_only':False}):
   meta,data=fixture();data.update(change)
   with self.assertRaises(ValueError):f.build_features(meta,data)
  for mutation in ('duplicate','dateorder','etfs'):
   meta,data=fixture()
   if mutation=='duplicate':meta['symbols'][0][1]='AAPL'
   elif mutation=='dateorder':meta['dates']=list(reversed(meta['dates']))
   else:meta['etfs']=meta['etfs'][:-1]
   with self.assertRaises(ValueError):f.build_features(meta,data)

if __name__=='__main__':unittest.main(verbosity=2)
