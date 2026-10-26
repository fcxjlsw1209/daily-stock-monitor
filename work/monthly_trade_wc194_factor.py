"""Pure annual CFO before three signed trade working-capital contributions.

No I/O, prices, fitting or portfolio work. The caller pins the normalized194
source admission and its independent complete closure. Latest report selection
precedes all validation. Original146 NI is required but absent from new ratios.
The residual is not GAAP CFO, full working capital, or organic cash generation.
"""
from collections import Counter
from decimal import InvalidOperation
from fractions import Fraction
import hashlib
import re
import numpy as np
import pandas as pd
import monthly_cash_quality146_factor as old

METHOD='current_annual_cash_before_trade_working_capital_v1'
DEFINITION='current_annual_trade_working_capital_signed_contributions_v1'
COMPONENTS=('accounts_receivable','inventories','accounts_payable')
EXTRA=('candidate_bounded','control_bounded','common_available')
DEPENDENCY_SHA256={'work/monthly_cash_quality146_factor.py':'6f5d819488c7952cb32894cf4b5a4f549c73adb570e949fd47c3c6ba2e561eea'}
CONCEPT_SHA256='380aae3cf18288dd668cefb3a9be94d0e9c90b76e8d085f6d59ff0ddafed9a83'
SOURCE_SCOPE_SHA256='f2dda7ce4d3d61084bfbbb6e3fabca2388250575588054a41bd33890dc4948be'
LABELS={
 'accounts_receivable':re.compile(r'^(?:trade\s+)?accounts? receivable(?:,?\s+net)?$',re.I),
 'inventories':re.compile(r'^inventor(?:y|ies)$',re.I),
 'accounts_payable':re.compile(r'^(?:trade\s+)?accounts? payable$',re.I)}


def _detail(reason,event=None):
 return dict(available=False,reason=reason,method=METHOD,
  current_event_id=None if event is None else event.get('event_id'),
  current_available_from=None if event is None else event.get('available_from'),
  current_period_end=None,prior_period_end=None,current_days=None,prior_days=None,
  currency_key=None,candidate_bounded=None,control_bounded=None)


def _gate(admission,definition,name):
 if type(admission) is not dict or set(admission)!={'admitted','reasons','definition'}:return name+'_admission_required'
 if admission['admitted'] is not True:return 'current_not_'+name+'_admitted'
 if admission['definition']!=definition:return 'wrong_'+name+'_definition'
 if type(admission['reasons']) is not list or admission['reasons']:return name+'_admitted_event_has_rejection_reasons'
 return None


def _decimal_fraction(text):
 old._require(old._text(text) and re.fullmatch(r'-?\d+(?:\.\d+)?',text),'Explicit plain decimal signed contribution required')
 return old._fraction(text)


def _literal_fraction(literal):
 old._require(old._text(literal),'Explicit signed displayed contribution required')
 text=re.sub(r'\s+',' ',literal.replace('/dollarsign','$').replace('\u2212','-')).strip()
 if re.fullmatch(r'\$?\s*[-\u2013\u2014]',text):return Fraction(0)
 match=re.fullmatch(r'\$?\s*(\()?\s*(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(\))?',text)
 old._require(match is not None and bool(match[1])==bool(match[3]) and not (match[1] and match[2].startswith('-')),
              'Ambiguous signed contribution literal')
 number=_decimal_fraction(match[2].replace(',',''))
 return -number if match[1] else number


def _statement_binding(cell,event):
 """Use original producer evidence; do not synthesize currency or scope fields."""
 loc=cell.get('locator');cfo_loc=event['fields']['current']['operating_cash_flow'].get('locator')
 statement=cell.get('statement_currency_evidence')
 old._require(type(loc) is dict and type(cfo_loc) is dict and type(statement) is dict,
              'Cashflow component statement evidence required')
 old._require(loc.get('original_cfo_locator')==cfo_loc and statement.get('original_cfo_locator')==cfo_loc,
              'Exact original CFO locator binding required')
 fmt=loc.get('format')
 old._require(fmt==cfo_loc.get('format')==statement.get('format'),'Cashflow statement formats differ')
 if fmt=='html':
  key='table_index_zero_based';position=loc.get(key)
  old._require(type(position) is int and position>=0 and position==cfo_loc.get(key)==statement.get(key),
               'Cashflow components must use same formal HTML table')
  old._require(loc.get('statement_segment')==cfo_loc.get('statement_segment')==statement.get('statement_segment'),
               'Cashflow statement segments differ')
  expected_scope='same formal cash-flow table; ISO unverified'
  row_position=loc.get('row_index_zero_based')
  old._require(type(row_position) is int and row_position>=0,'Explicit cashflow component HTML row required')
 elif fmt=='pdf':
  key='page_1based';position=loc.get(key)
  old._require(type(position) is int and position>0 and position==cfo_loc.get(key)==statement.get(key),
               'Cashflow components must use same formal PDF page')
  expected_scope='same formal cash-flow page; ISO unverified'
  row_position=loc.get('line_1based')
  old._require(type(row_position) is int and row_position>0,'Explicit cashflow component PDF line required')
 else:raise ValueError('Unknown cashflow source format')
 old._require(loc.get('period_end')==event['periods']['current']['period_end'],'Component locator annual period differs')
 row_currency=cell.get('currency_literal');explicit=cell.get('statement_currency_literal')
 old._require(row_currency in (None,'$','/dollarsign') and explicit in (None,'$') and cell.get('currency_iso') is None,
              'Conflicting dollar currency evidence')
 old._require('currency_scope' not in cell or cell['currency_scope']==expected_scope,'Conflicting statement currency scope')
 old._require('literal' not in statement or statement['literal']=='$','Conflicting statement literal')
 explicit_path=explicit=='$' and cell.get('currency_scope')==expected_scope
 rows_path=False
 if statement.get('literal')=='$':
  rows=statement.get('rows')
  old._require(fmt=='html' and type(rows) is list and bool(rows),'Same-table dollar rows required')
  for row in rows:
   old._require(type(row) is dict and type(row.get('row_index_zero_based')) is int and row['row_index_zero_based']>=0
                and type(row.get('row_text')) is str and '$' in row['row_text']
                and type(row.get('row_html_sha256')) is str and re.fullmatch('[0-9a-f]{64}',row['row_html_sha256']),
                'Invalid same-table dollar row evidence')
  rows_path=True
 old._require(explicit_path or rows_path,'Verified native statement currency evidence required')
 return (fmt,position,row_position)


def _trade_contributions(event):
 cells=event.get('trade_working_capital');evidence=event.get('trade_working_capital_source_evidence')
 old._require(type(cells) is dict and set(cells)==set(COMPONENTS) and type(evidence) is dict,'All three component cells/evidence required')
 for key in ('event_id','symbol','release_date','available_from','source'):
  old._require(evidence.get(key)==event.get(key),'Working-capital evidence metadata differs: '+key)
 old._require(evidence.get('inherited_admitted') is True and evidence.get('extraction_verified') is True
              and evidence.get('status')=='verified' and type(evidence.get('hold_reasons')) is list
              and not evidence['hold_reasons'],'Verified inherited component extraction required')
 raw=evidence.get('fields')
 old._require(type(raw) is dict and set(raw)==set(COMPONENTS),'Exactly three original producer fields required')
 values=[];locators=[]
 cash_scale=old._fraction(event['fields']['current']['operating_cash_flow']['scale'])
 for key in COMPONENTS:
  cell=cells[key];original=raw[key]
  old._require(type(cell) is dict and type(original) is dict and all(k in cell and cell[k]==v for k,v in original.items()),
               'Normalized component changed original evidence: '+key)
  old._require(cell.get('source')==event['source'] and cell.get('period')==event['periods']['current']
               and cell.get('available_from')==event['available_from'] and cell.get('fact_admitted') is False,
               'Component source/period/clock/raw fact differs: '+key)
  old._require(cell.get('component')==key,'Component identity differs')
  label=cell.get('row_label');row=cell.get('row_text')
  old._require(type(label) is str and LABELS[key].fullmatch(label),'Combined or nontrade component excluded: '+key)
  old._require(type(row) is str and hashlib.sha256(row.encode()).hexdigest()==cell.get('row_text_sha256'),
               'Component row text hash differs')
  locators.append(_statement_binding(cell,event))
  signed=_decimal_fraction(cell['signed_value_decimal']);scale=_decimal_fraction(cell['scale'])
  old._require(scale>0 and scale==cash_scale,'Same-report cashflow scales differ')
  old._require(_literal_fraction(cell['signed_literal'])==signed,'Signed displayed contribution differs from decimal')
  values.append(signed*scale)
 old._require(len(set(locators))==3,'Distinct operating-cashflow component rows required')
 return tuple(values)


def evaluate_report(event,signal):
 """Validate one already-selected report, then use a common mask for both arms."""
 old._date(signal)
 if type(event) is not dict:return _detail('invalid_event')
 detail=_detail('unavailable',event)
 for name,definition in (('cash_quality',old.DEFINITION),('trade_working_capital',DEFINITION)):
  reason=_gate(event.get(name+'_admission'),definition,name)
  if reason:
   detail['reason']=reason;return detail
 legacy=old.evaluate_report(event,signal)
 for key in ('current_period_end','prior_period_end','current_days','prior_days','currency_key'):detail[key]=legacy[key]
 if not legacy['available']:
  detail['reason']=legacy['reason'];return detail
 try:
  contributions=_trade_contributions(event)
  cash=old._amount(event['fields']['current']['operating_cash_flow'],event,'current')
  assets_now=old._amount(event['fields']['current']['total_assets'],event,'current')
  assets_prior=old._amount(event['fields']['comparison']['total_assets'],event,'comparison')
  average_assets=(assets_now+assets_prior)/2
  working_capital=sum(contributions,Fraction(0))
  candidate,control=(cash-working_capital)/average_assets,cash/average_assets
 except (KeyError,TypeError,ValueError,ArithmeticError,InvalidOperation):
  detail['reason']='invalid_or_missing_trade_working_capital_evidence';return detail
 detail.update(available=True,reason='available',candidate_bounded=old.bounded(candidate),control_bounded=old.bounded(control))
 return detail


def _cell(events,signal):
 day=old._date(signal);known={}
 for event in events:
  old._require(type(event) is dict and old._text(event.get('event_id')),'Event identity required')
  available=old._date(event.get('available_from'))
  if available>day:continue
  key=event['event_id']
  if key in known and known[key]!=event:return _detail('conflicting_event_id')
  known[key]=event
 if not known:return _detail('current_missing_event')
 latest=max(e['available_from'] for e in known.values());selected=[e for e in known.values() if e['available_from']==latest]
 if len(selected)!=1:return _detail('current_ambiguous_latest_event')
 return evaluate_report(selected[0],signal)


def build_features(meta,admitted_dataset):
 """Same-support current annual CFO and CFO less signed trade contributions."""
 old._require(type(admitted_dataset) is dict and type(admitted_dataset.get('schema_version')) is int
              and admitted_dataset['schema_version']==1 and admitted_dataset.get('method')==METHOD
              and admitted_dataset.get('conditional_research_only') is True,'Explicit conditional194 dataset required')
 events=admitted_dataset.get('events');old._require(type(events) is list,'Complete event list required')
 by_symbol={}
 for event in events:
  old._require(type(event) is dict and old._text(event.get('symbol')),'Declared event symbol required')
  by_symbol.setdefault(event['symbol'],[]).append(event)
 dates=pd.DatetimeIndex(meta['dates'])
 old._require(dates.tz is None and not dates.hasnans and dates.is_unique and dates.is_monotonic_increasing
              and dates.equals(dates.normalize()),'Sorted unique naive signal dates required')
 old._require(len(meta['symbols'])==len(dates),'Original signal/symbol axes required')
 etfs=meta['etfs']
 old._require(len(etfs)==17 and len(set(etfs))==17 and all(old._text(e) for e in etfs),'Original17 ETFs required')
 arrays={key:np.full((len(dates),49),np.nan) for key in EXTRA[:-1]};arrays['common_available']=np.zeros((len(dates),49),dtype=bool)
 cells=[]
 for i,(day,names) in enumerate(zip(dates,meta['symbols'])):
  old._require(len(names)==49 and len(set(names))==49 and set(etfs)<=set(names) and all(old._text(n) for n in names),
               'Original32 stocks plus17 ETFs required')
  signal=str(day.date())
  for j,symbol in enumerate(names):
   detail=_detail('etf_not_applicable') if symbol in etfs else _cell(by_symbol.get(symbol,[]),signal)
   if detail['available']:
    arrays['common_available'][i,j]=True
    for key in EXTRA[:-1]:arrays[key][i,j]=detail[key]
   cells.append(dict(signal_index=i,pool_index=j,signal_date=signal,symbol=symbol,**detail))
 audit=dict(original_arrays_unchanged=True,original_months=len(dates),original_width=49,
  common_available_cells=int(arrays['common_available'].sum()),reasons=dict(Counter(c['reason'] for c in cells)),cells=cells,
  new_model_fits=0,returns_or_nav_computed=False,latest_event_before_validity_filter=True,same_current_release_for_all_cells=True,
  both_admissions_required=True,original_ni_gate_preserved=True,net_income_used_in_new_ratios=False,
  comparison_flow_values_used=False,annual_duration_normalization=False,
  original_auxiliary_states_and_core_coefficients_preserved=True,
  source_verification='caller_verified_trade_working_capital_source_review_and_complete_hash_closure_required',
  arithmetic_method='exact_scaled_decimal_fraction_ratios_then_single_bounded_float',
  currency_scope='same_report_dollar_literal_assumption_not_ISO_certification',
  interpretation='annual_as_reported_CFO_less_three_signed_trade_contributions_not_GAAP_or_organic_CFO',
  all_three_signed_components_required=True,component_signs_not_inverted=True,component_sum_before_bounded_mapping=True)
 return dict(arrays=arrays,audit=audit)
