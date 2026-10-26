"""Independent round146 preparation oracle; no producer imports or import-time data I/O.

Exact integer decimal alignment and closed-form bounded ratios independently
check the producer's rational-ratio construction. Real-file auditing requires an
explicit registered PLAN SHA and completed preparation. Pure oracles import only
the pinned old independent selector helper, which never loads a market cache.
"""
from pathlib import Path
from bisect import bisect_right
import hashlib
import importlib.util
import sys
from datetime import date, timedelta
from decimal import Decimal
from collections import Counter
import argparse
import json
import math
import re
import numpy as np
import pandas as pd
import monthly_latent_state134_prepare_review as old

ROOT=Path(__file__).resolve().parents[1]
STUDY='outputs/monthly_cash_quality146'
LOCAL=old.LOCAL
NAMES=('incumbent_local67','earnings_only','cash_flow_only','low_accrual_quality')
METHOD='current_annual_cash_quality_v1'
DEFINITION='consolidated_gaap_cfo_net_income_average_assets_v1'
EXTRA=('earnings_bounded','cash_flow_bounded','low_accrual_bounded','common_available')
SIDES=('current','comparison')
FIELDS=('operating_cash_flow','net_income','total_assets')
REQUIRED=(('current','operating_cash_flow'),('current','net_income'),('current','total_assets'),('comparison','total_assets'))
CHECKS=('original27_arrays_byte_identical','all_features_independently_reconstructed',
 'all_four_score_matrices_and_ordered_choices_independently_verified',
 'one_episode_and_prefix_isolation_verified','zero_and_missing_exact_fallback')
ATOL=1e-12
INPUTS=('proposal','meta','policy','original_orders','prices','sessions','protected_manifest','admitted_dataset','source_admission')
CACHE_FILES=('DATA.npz','SCORES.npz','META.json','FEATURE_AUDIT.json','PRETRAIN_ORDERS.json','PRICE_PARITY.json')
CALENDAR_SHA='4b8f32ad8a555d6537d7cf3bd0f831e64fbb2de26d033c8daaf68590fcd4a6ea'
PROPOSAL_SHA='585fea23faef22c724b7977273f3d3c5994c673198c0e88115f12288accdf51a'
require=old.require


def source_files(root=None):
 root=ROOT if root is None else Path(root).resolve()
 return sorted(set([root/'work/monthly_cash_quality146_prepare_review.py',root/'work/test_monthly_cash_quality146_prepare_review.py']+old.source_files(root)))


def text(value):return type(value)is str and bool(value) and value.strip()==value


def day(value):
 require(type(value)is str,'Explicit ISO date')
 parsed=date.fromisoformat(value);require(parsed.isoformat()==value,'Canonical ISO date')
 return parsed


def decimal_integer(value):
 """Return an exact integer coefficient and base-10 exponent, without division."""
 require(text(value) and re.fullmatch(r'-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',value),'Explicit finite decimal string')
 d=Decimal(value);require(d.is_finite(),'Finite decimal required')
 t=d.as_tuple();coefficient=0
 for digit in t.digits:coefficient=10*coefficient+digit
 return (-coefficient if t.sign else coefficient),t.exponent


def scaled_integer(cell,event,side):
 require(type(cell)is dict,'Required cell missing')
 require(cell.get('source')==event['source'] and cell.get('period')==event['periods'][side]
         and cell.get('available_from')==event['available_from'],'Same-source period and availability')
 require(cell.get('currency_literal')=='$','Same-report dollar literal')
 a,e=decimal_integer(cell['value_decimal']);b,f=decimal_integer(cell['scale'])
 require(b>0,'Positive scale')
 return a*b,e+f


def bounded_from_cells(values):
 """CFO, NI, current assets, previous assets -> three exact closed-form ratios."""
 common=min(exponent for coefficient,exponent in values)
 cash,earnings,current,previous=[coefficient*10**(exponent-common) for coefficient,exponent in values]
 require(current>0 and previous>0,'Positive asset endpoints')
 assets=current+previous
 numerators=(2*earnings,2*cash,2*(cash-earnings))
 return tuple(n/(assets+abs(n)) for n in numerators)


def detail(reason,event=None):
 return dict(available=False,reason=reason,method=METHOD,
  current_event_id=None if event is None else event.get('event_id'),
  current_available_from=None if event is None else event.get('available_from'),
  current_period_end=None,prior_period_end=None,current_days=None,prior_days=None,currency_key=None,
  earnings_bounded=None,cash_flow_bounded=None,low_accrual_bounded=None)


class InvalidReport(Exception):pass


def check(condition,reason):
 if not condition:raise InvalidReport(reason)


def period(value,side):
 try:
  check(type(value)is dict,side+'_invalid_period_dates')
  start,end=day(value['period_start']),day(value['period_end'])
 except (KeyError,TypeError,ValueError):raise InvalidReport(side+'_invalid_period_dates') from None
 length=(end-start).days+1
 check(330<=length<=400,side+'_nonannual_duration')
 check(type(value.get('days_inclusive'))is int and value['days_inclusive']==length,side+'_duration_mismatch')
 return start,end,length


def report_oracle(event,signal):
 signal_day=day(signal);out=detail('unavailable',event)
 try:
  admission=event.get('cash_quality_admission')
  check(type(admission)is dict and set(admission)=={'admitted','reasons','definition'},'cash_quality_admission_required')
  check(admission['admitted']is True,'current_not_cash_quality_admitted')
  check(admission['definition']==DEFINITION,'wrong_cash_quality_definition')
  check(type(admission['reasons'])is list and not admission['reasons'],'admitted_event_has_rejection_reasons')
  check(type(event.get('inherited_rejection_codes'))is list and not event['inherited_rejection_codes'],'inherited_rejection_not_cleared')
  check(all(text(event.get(k)) for k in ('event_id','symbol')) and event['symbol']in ('AAPL','AMD','MSFT','NVDA'),'unsupported_or_missing_identity')
  identity=event.get('identity')
  check(type(identity)is dict and all(text(identity.get(k)) for k in ('issuer_id','security_id')),'missing_identity')
  check(identity['security_id']==event['symbol'],'security_symbol_mismatch')
  source=event.get('source')
  check(type(source)is dict and text(source.get('path')) and type(source.get('sha256'))is str and bool(re.fullmatch('[0-9a-f]{64}',source['sha256'])),'missing_source_reference')
  try:
   available,released=day(event['available_from']),day(event['release_date'])
   identity_start,identity_end=day(identity['identity_valid_from']),day(identity['identity_valid_through'])
  except (KeyError,TypeError,ValueError):raise InvalidReport('invalid_dates_or_identity_bounds') from None
  check('current_release_available_from' not in event or event['current_release_available_from']==event['available_from'],'current_release_availability_mismatch')
  check(available<=signal_day,'report_not_yet_available')
  check(released<available,'availability_not_after_release')
  check(identity_start==available,'identity_start_not_current_availability')
  periods,fields=event.get('periods'),event.get('fields')
  check(type(periods)is dict and set(periods)==set(SIDES),'two_annual_periods_required')
  check(type(fields)is dict and set(fields)==set(SIDES) and all(type(fields[s])is dict and set(fields[s])==set(FIELDS) for s in SIDES),'raw_field_schema_required')
  start,end,length=period(periods['current'],'current')
  pstart,pend,plength=period(periods['comparison'],'comparison')
  out.update(current_period_end=str(end),prior_period_end=str(pend),current_days=length,prior_days=plength,currency_key='same-report-dollar:'+event['event_id'])
  check(end<=released,'release_before_period_end')
  check(pend+timedelta(days=1)==start,'noncontiguous_annual_periods')
  check(identity_end==end+timedelta(days=550),'identity_end_not_causal_expiry')
  check((signal_day-end).days<=550,'stale_current_report')
  check(identity_start<=signal_day<=identity_end,'identity_outside_interval')
  amounts=[]
  for side,field in REQUIRED:
   check(fields[side][field]is not None,'missing_'+side+'_'+field)
   try:amounts.append(scaled_integer(fields[side][field],event,side))
   except (KeyError,TypeError,ValueError,ArithmeticError):raise InvalidReport('invalid_'+side+'_'+field) from None
  check(amounts[2][0]>0 and amounts[3][0]>0,'nonpositive_asset_endpoint')
  signals=bounded_from_cells(amounts)
  out.update(available=True,reason='available',**dict(zip(EXTRA[:3],signals)))
 except InvalidReport as exc:out['reason']=str(exc)
 return out


def cell_oracle(events,signal):
 signal_day=day(signal);known={}
 require(type(events)is list,'Event list')
 for event in events:
  require(type(event)is dict and text(event.get('event_id')),'Explicit event identity')
  if day(event.get('available_from'))>signal_day:continue
  key=event['event_id']
  if key in known and known[key]!=event:return detail('conflicting_event_id')
  known[key]=event
 if not known:return detail('current_missing_event')
 latest=max(day(e['available_from']) for e in known.values())
 selected=[e for e in known.values() if day(e['available_from'])==latest]
 if len(selected)!=1:return detail('current_ambiguous_latest_event')
 return report_oracle(selected[0],signal)


def features_oracle(meta,dataset):
 require(type(dataset)is dict and type(dataset.get('schema_version'))is int and dataset['schema_version']==1 and dataset.get('method')==METHOD and type(dataset.get('events'))is list,'New cash-quality dataset schema')
 dates=pd.DatetimeIndex(meta['dates'])
 require(not dates.hasnans and dates.tz is None and dates.is_unique and dates.is_monotonic_increasing and dates.equals(dates.normalize()),'Canonical signal dates')
 require(len(dates)==len(meta['symbols']),'Aligned signal axes')
 etfs=meta['etfs'];require(len(etfs)==len(set(etfs))==17 and all(text(s) for s in etfs),'Original seventeen ETFs')
 grouped={}
 for event in dataset['events']:
  require(type(event)is dict and text(event.get('symbol')),'Event symbol')
  grouped.setdefault(event['symbol'],[]).append(event)
 arrays={key:np.full((len(dates),49),np.nan) for key in EXTRA[:3]};arrays[EXTRA[3]]=np.zeros((len(dates),49),bool);cells=[]
 for i,(signal,names) in enumerate(zip(dates,meta['symbols'])):
  require(len(names)==len(set(names))==49 and set(etfs)<=set(names) and all(text(s) for s in names),'Original forty-nine candidates')
  for j,symbol in enumerate(names):
   value=detail('etf_not_applicable') if symbol in etfs else cell_oracle(grouped.get(symbol,[]),str(signal.date()))
   cells.append(dict(signal_index=i,pool_index=j,signal_date=str(signal.date()),symbol=symbol,**value))
   if value['available']:
    arrays['common_available'][i,j]=True
    for key in EXTRA[:3]:arrays[key][i,j]=value[key]
 return dict(arrays=arrays,cells=cells)


def score_oracle(arrays,meta,earnings,cash_flow,low_accrual,n=None):
 helper=old.original_helper();x=helper.original_prefix(arrays,meta,n);n=x['n'];shape=x['core'].shape
 hands=[np.asarray(h)[:n] for h in (earnings,cash_flow,low_accrual)]
 for hand in hands:
  require(hand.shape==shape and hand.dtype.kind in 'fiu' and not np.isinf(hand).any(),'Bounded hand shape/type')
  require((np.abs(hand[np.isfinite(hand)])<=1).all(),'Bounded hand range')
  for i,names in enumerate(meta['symbols'][:n]):
   require(all(not np.isfinite(hand[i,j]) for j,s in enumerate(names) if s in meta['etfs']),'ETFs have no feature')
 require(all(np.array_equal(np.isfinite(hands[0]),np.isfinite(h)) for h in hands[1:]),'One common availability mask')
 scores={};choices={};deltas={}
 for k,name in enumerate(NAMES):
  pre=x['local'].copy();delta=np.zeros(shape)
  if k:
   hand=hands[k-1];active=x['eligible'] & np.isfinite(hand) & (hand!=0)
   delta[active]=.01*hand[active]
   pre[active]+=delta[active]
  choice,final,_=helper.episode(pre,x)
  scores[name]=final;choices[name]=choice;deltas[name]=delta
 return scores,choices,deltas


def verify_original_arrays(original,prepared):
 require(len(original)==27 and set(prepared)==set(original)|set(EXTRA),'Original 27 plus four new arrays')
 for key,value in original.items():
  candidate=prepared[key]
  require(value.shape==candidate.shape and value.dtype==candidate.dtype and value.tobytes()==candidate.tobytes(),'Original array changed: '+key)


def verify_audit(saved,expected,*,dataset_sha=None,admission_sha=None):
 require(type(saved.get('new_model_fits'))is int and saved['new_model_fits']==0 and saved.get('returns_or_nav_computed')is False,'No new fit or returns')
 for key in ('original_arrays_unchanged','original_auxiliary_states_and_core_coefficients_preserved','latest_event_before_validity_filter','same_current_release_for_all_cells'):
  require(saved.get(key)is True,'Required feature audit claim: '+key)
 for key in ('annual_duration_normalization','comparison_flow_values_used'):require(saved.get(key)is False,'Forbidden feature transformation: '+key)
 for key,value in (
  ('arithmetic_method','exact_fraction_scaled_amounts_average_assets_ratios_then_float64_bounded_mapping'),
  ('source_verification','caller_verified_new_field_specific_admission_and_source_hashes_required'),
  ('currency_scope','same_report_dollar_literal_assumption_not_ISO_certification')):
  require(saved.get(key)==value,'Feature audit method: '+key)
 if dataset_sha is not None or admission_sha is not None:
  require(dataset_sha and admission_sha and saved.get('admitted_dataset_sha256')==dataset_sha and saved.get('source_admission_sha256')==admission_sha,'Source audit bindings')
 require(saved.get('common_available_cells')==int(expected['arrays']['common_available'].sum()),'Common-mask count')
 require(saved.get('original_months')==len(expected['cells'])//49 and saved.get('original_width')==49,'Feature audit axes')
 require(saved.get('reasons')==dict(Counter(c['reason'] for c in expected['cells'])),'Full reason census')
 require(len(saved.get('cells',[]))==len(expected['cells']),'Every feature cell audited')
 for actual,wanted in zip(saved['cells'],expected['cells']):
  require(set(actual)==set(wanted),'Exact feature audit cell schema')
  for key,value in wanted.items():
   require(type(actual.get(key))is type(value),'Feature audit type: '+key)
   if type(value)is float:require(math.isclose(actual[key],value,abs_tol=ATOL,rel_tol=0),'Feature audit value: '+key)
   else:require(actual[key]==value,'Feature audit value: '+key)


def ordered_orders(meta,scores,choices):
 return {name:[dict(date=day,signal_date=day,end=meta['ends'][i],symbols=[meta['symbols'][i][j] for j in choice],
  scores=[float(scores[name][i,j]) for j in choice]) for i,(day,choice) in enumerate(zip(meta['dates'],choices[name]))] for name in NAMES}


def verify_scores_and_orders(actual_scores,actual_orders,meta,expected_scores,expected_choices):
 require(set(actual_scores)==set(actual_orders)==set(NAMES),'Exactly four saved arms')
 orders=ordered_orders(meta,expected_scores,expected_choices)
 for name in NAMES:
  require(actual_scores[name].shape==expected_scores[name].shape and actual_scores[name].dtype==expected_scores[name].dtype,'Full score matrix axes and dtype')
  np.testing.assert_allclose(actual_scores[name],expected_scores[name],rtol=0,atol=ATOL)
  require(len(actual_orders[name])==len(orders[name]),'Every month ordered')
  for actual,wanted in zip(actual_orders[name],orders[name]):
   require(set(actual)==set(wanted) and all(actual[k]==wanted[k] for k in ('date','signal_date','end','symbols')),'Ordered holding mismatch')
   np.testing.assert_allclose(actual['scores'],wanted['scores'],rtol=0,atol=ATOL)


def source_gate_oracle(event,sessions):
 """Check source admission only; signal-day validity is a separate later gate.

 No ratios are needed here. In particular, source admission need not promise
 that a report is usable before its 550-day expiry at any monthly signal.
 """
 try:
  if event.get('prior_field_admitted')is not True or event.get('inherited_rejection_codes') or event.get('metadata_gaps'):return False
  released,available=day(event['release_date']),day(event['available_from'])
  i=bisect_right(sessions,str(released))
  if i>=len(sessions) or sessions[i]!=str(available):return False
  identity=event['identity']
  if identity['security_id']!=event['symbol'] or not identity['issuer_id'] or identity['identity_valid_from']!=str(available):return False
  endpoints={}
  for side in SIDES:
   p=event['periods'][side];start,end=day(p['period_start']),day(p['period_end']);length=(end-start).days+1
   if not 330<=length<=400 or type(p['days_inclusive'])is not int or p['days_inclusive']!=length:return False
   endpoints[side]=start,end
  if endpoints['comparison'][1]+timedelta(days=1)!=endpoints['current'][0] or endpoints['current'][1]>released:return False
  if identity['identity_valid_through']!=str(endpoints['current'][1]+timedelta(days=550)):return False
  for side,field in REQUIRED:
   cell=event['fields'][side][field]
   if type(cell)is not dict:return False
   if cell['source']!=event['source'] or cell['available_from']!=event['available_from'] or cell['period']!=event['periods'][side]:return False
   if cell['currency_literal']!='$':return False
   if cell.get('fact_admitted')is not False or cell.get('currency_iso')not in (None,'USD'):return False
   value,scale=Decimal(cell['value_decimal']),Decimal(cell['scale'])
   if not value.is_finite() or not scale.is_finite() or scale<=0 or (field=='total_assets' and value<=0):return False
   if cell['locator']['period_end']!=event['periods'][side]['period_end']:return False
   if field=='net_income' and cell['locator']['ni_scope']!='indirect_method_consolidated_start':return False
  return True
 except (KeyError,ValueError,TypeError,ArithmeticError):return False


def verify_admission(reader,plan,pins,sessions):
 refs=plan['inputs'];receipt=reader.json(refs['source_admission']['path']);dataset=reader.json(refs['admitted_dataset']['path'])
 require(receipt.get('stage')=='conditional_cash_quality_source_admission' and receipt.get('passed')is True,'Conditional cash-quality admission required')
 require(receipt.get('method')==dataset.get('method')==METHOD and receipt['data_sha256']==refs['admitted_dataset']['sha256'],'Bound cash-quality dataset')
 policyref=plan['source_policy'];policypath=reader.path(policyref['path'])
 require(reader.sha(policypath)==policyref['sha256']==receipt['policy_sha256']==dataset['policy_sha256']==pins.get(str(policypath)),'Source policy binding')
 policy=reader.json(policypath)
 require(policy.get('schema_version')==1 and policy['method']==METHOD and policy['definition']==DEFINITION and policy['expected_events']==44,'Source policy definition')
 require(dataset.get('schema_version')==1 and dataset.get('conditional_research_only')is True,'Conditional dataset schema')
 require(receipt.get('schema_version')==1 and receipt.get('events')==44 and receipt.get('currency_iso_not_newly_certified')is True,'Admission scope and census')
 implementation=reader.path('work/cash_quality_source_admission_v1.py')
 require(policy['runtime']==sys.version and reader.sha(implementation)==policy['implementation_sha256']==pins.get(str(implementation)),'Source admission implementation and runtime')
 for key in ('preserve_all_inherited_holds','same_report_literal_currency_assumption'):require(policy.get(key)is True,'Required source assumption')
 for key in ('use_comparison_flows','financial_ratios_permitted','strategy_registration_permitted'):require(policy.get(key)is False,'Source-only policy scope')
 require(policy['annual_days']==[330,400] and policy['max_age_days']==550,'Source time policy')
 for key in ('financial_ratios','factor_calculations','portfolio_calculations','return_calculations','network_requests','observer_changes'):
  require(type(receipt.get(key))is int and receipt[key]==0,'Admission pre-factor isolation')
 for key in ('first_public_version_verified','complete_correction_history_verified','certified_daily_security_master'):require(receipt.get(key)is False,'No historical certification')
 require(receipt.get('original_raw_events_preserved')is True and receipt.get('original_rejections_preserved')is True,'Source preservation claims')
 source_refs=policy['inputs'];require(set(source_refs)=={'union_plan','union_manifest','union_data','union_independent','accounting','sessions','concept','tests'},'Exact source admission inputs')
 require(receipt['source_hashes']=={ref['path']:ref['sha256'] for ref in source_refs.values()},'Exact source admission closure')
 objects={}
 for name,ref in source_refs.items():
  path=reader.path(ref['path']);require(reader.sha(path)==ref['sha256']==pins.get(str(path)),'Registered source evidence: '+name);objects[name]=reader.json(path)
 require(objects['tests'].get('passed')is True and objects['tests'].get('source_sha256')==policy['implementation_sha256'],'Current admission synthetic tests')
 require(objects['accounting'].get('stage')=='source_only_field_scoped_accounting_review' and objects['accounting'].get('source_only')is True,'Source-only accounting review')
 require(objects['sessions']['sessions']==sessions,'Shared source/session calendar')
 raw=objects['union_data'];manifest=objects['union_manifest'];union_plan=objects['union_plan'];review=objects['union_independent']
 require(raw['plan_sha256']==manifest['plan_sha256']==source_refs['union_plan']['sha256'],'Raw union plan binding')
 require(manifest['output_hashes']['RAW_FACTS.json']==source_refs['union_data']['sha256'],'Raw union data binding')
 for group in ('source_hashes','source_code_hashes'):
  require(all(pins.get(k)==v for k,v in reader.verify(union_plan[group]).items()),'Complete registered raw union provenance')
 reviewpins={}
 for group in ('input_hashes','source_hashes'):
  if group in review:reviewpins.update({str(reader.path(k)):v for k,v in review[group].items()})
 require(review.get('passed')is True and all(reviewpins.get(str(reader.path(source_refs[k]['path'])))==source_refs[k]['sha256'] for k in ('union_plan','union_manifest','union_data')),'Completed independent raw union review')
 originals={e['event_id']:e for e in raw['events']};events=dataset['events'];accounting={e['event_id']:e for e in objects['accounting']['events']}
 require(len(originals)==len(raw['events'])==len(events)==len({e['event_id'] for e in events})==len(accounting)==44,'Full unique 44 event census')
 require(set(originals)==set(accounting)=={e['event_id'] for e in events},'Exact event identities')
 require([e['event_id'] for e in events]==[e['event_id'] for e in raw['events']],'Raw event order preserved')
 admitted=[]
 for event in events:
  key=event['event_id'];require({k:v for k,v in event.items() if k!='cash_quality_admission'}==originals[key],'Raw event preservation')
  require(all(event.get(k)is False for k in ('fact_admitted','model_admitted','strategy_admitted')),'Raw admission flags retained')
  gate=event['cash_quality_admission'];require(set(gate)=={'admitted','reasons','definition'} and gate['definition']==DEFINITION,'New admission gate schema')
  allowed=source_gate_oracle(event,sessions)
  require(gate['admitted']is allowed and type(gate['reasons'])is list and (not gate['reasons'])==allowed,'Independent admission decision')
  provenance=event['union_provenance']
  require(provenance['accounting_review_reference']==source_refs['accounting'] and provenance['accounting_event_review']==accounting[key],'Exact accounting provenance')
  require(accounting[key]['inherited_rejection_codes']==event['inherited_rejection_codes'] and accounting[key]['inherited_rejections_unchanged']is True,'Inherited holds retained')
  if allowed:admitted.append(key)
 require(receipt['admitted_event_ids']==admitted and receipt['admitted_events']==len(admitted) and receipt['held_events']==44-len(admitted),'Admission counts and IDs')
 require(receipt['holds']==[dict(event_id=e['event_id'],**e['cash_quality_admission']) for e in events if not e['cash_quality_admission']['admitted']],'Every admission hold retained')
 return dataset


def synthetic_receipt(result,root=None):
 """Serialize only a completed synthetic run and read-only source metadata."""
 root=ROOT if root is None else Path(root).resolve()
 sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files(root)}
 reviewed=[root/'work'/name for name in ('monthly_cash_quality146_runner.py','monthly_cash_quality146_factor.py',
  'monthly_forward_guidance132_runner.py','cash_quality_source_admission_v1.py','monthly_cash_quality146_proposal.json')]
 return dict(schema_version=1,stage='synthetic_prepare_review_only',passed=result.wasSuccessful(),
  tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
  runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),
  source_hashes=sources,reviewed_contract_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in reviewed},
  method=METHOD,arms=list(NAMES),features=list(EXTRA),required_checks=list(CHECKS),cache_files=list(CACHE_FILES),
  required_inputs=list(INPUTS),required_source_policy_reference=['path','sha256'],
  actual_preparation_checked=False,actual_factors_calculated=0,actual_returns_calculated=0,
  producer_modules_imported=False,source_gate_separate_from_signal_day_validity=True,
  actual_api='./run_local.sh work/monthly_cash_quality146_prepare_review.py --plan <PLAN.json> --plan-sha <SHA256> --prepare-complete',
  actual_output=STUDY+'/INDEPENDENT_PREPARE.json',
  pending_requirements=['Root must supply frozen round146 PLAN SHA after registration and completed preparation.',
   'PLAN.hashes must include source_policy, its eight inputs, source admission implementation, union provenance, raw64, original cache and independent source closure.'],
  scope_limits=['Source admission decisions are checked independently as booleans; saved hold reasons are preserved exactly from dataset into receipt, not reworded.',
   'This receipt certifies only synthetic tests and the reviewed source contract, not real data or strategy results.'])


def verify_prefixes(arrays,meta,dataset,expected,scores,choices):
 for n in range(len(meta['dates'])+1):
  subset=dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n])
  historical=dict(dataset,events=[e for e in dataset['events'] if n and e['available_from']<=meta['dates'][n-1]])
  prefix=features_oracle(subset,historical)
  require(prefix['cells']==expected['cells'][:n*49],'Source prefix isolation')
  for key in EXTRA:np.testing.assert_array_equal(prefix['arrays'][key],expected['arrays'][key][:n])
  ss,cc,_=score_oracle(arrays,meta,*(prefix['arrays'][k] for k in EXTRA[:3]),n=n)
  for name in NAMES:require(cc[name]==choices[name][:n] and ss[name].tobytes()==scores[name][:n].tobytes(),'Score and holding prefix isolation')
 for fill in (np.nan,0.):
  hand=np.full((len(meta['dates']),len(meta['symbols'][0])),np.nan)
  for i,names in enumerate(meta['symbols']):
   for j,symbol in enumerate(names):
    if symbol not in meta['etfs']:hand[i,j]=fill
  ss,cc,_=score_oracle(arrays,meta,hand,hand,hand)
  for name in NAMES[1:]:require(ss[name].tobytes()==scores[NAMES[0]].tobytes() and cc[name]==choices[NAMES[0]],'Exact zero/missing incumbent fallback')


def audit(root,plan_path,plan_sha):
 require(type(plan_sha)is str and re.fullmatch('[0-9a-f]{64}',plan_sha),'Explicit PLAN SHA required before any file read')
 reader=old.Reader(root);directory=reader.path(STUDY)
 require(reader.path(plan_path)==directory/'PLAN.json','Dedicated round146 PLAN')
 require(reader.sha(plan_path)==plan_sha,'Frozen PLAN hash');plan=reader.json(plan_path)
 require(plan.get('round')==146 and plan.get('status')=='REGISTERED' and plan.get('arms')==list(NAMES) and plan.get('primary_challenger')==NAMES[3],'Round146 frozen arms')
 require(reader.path(plan['output_directory'])==directory and plan['dose']==.01,'Output and dose policy')
 require(plan['runtime']==dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),'Pinned runtime')
 require(not any((directory/name).exists() for name in ('TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json','WALKFORWARD.json','FROZEN.json','RESULTS.json')),'Preparation audit must precede outcomes')
 pins=reader.verify(plan['hashes']);require(set(plan['inputs'])==set(INPUTS),'Named inputs')
 inputs={}
 for name,ref in plan['inputs'].items():
  path=reader.path(ref['path']);require(reader.sha(path)==ref['sha256']==pins.get(str(path)),'Bound input: '+name);inputs[name]=path
 require(reader.sha(inputs['proposal'])==PROPOSAL_SHA,'Reviewed final prospective proposal')
 reader.json(inputs['proposal'])
 for key in ('actual_model_fits_before_registration','actual_feature_calculations_before_registration','actual_returns_before_registration'):
  require(type(plan.get(key))is int and plan[key]==0,'Prospective registration')
 require(plan['tolerances']['feature_atol']==ATOL and plan['tolerances']['scores_atol']==ATOL and plan['tolerances']['scores_rtol']==0,'Exact preparation tolerances')
 sources={str(path):reader.sha(path) for path in source_files(root)}
 require(reader.verify(plan['independent_prepare_source_hashes'])==sources and all(pins.get(k)==v for k,v in sources.items()),'Independent source closure')
 oldplan=reader.json(LOCAL+'/PLAN.json');raw=reader.verify(plan['raw_source_hashes'])
 require(raw==reader.verify(oldplan['raw_source_hashes']) and len(raw)==64 and all(pins.get(k)==v for k,v in raw.items()),'Original raw64 unchanged')
 original_cache=reader.verify(reader.json(LOCAL+'/CACHE_HASHES.json'),reader.path(LOCAL))
 require(all(pins.get(k)==v for k,v in original_cache.items()),'Original cache bindings')
 for name,file in (('meta','META.json'),('policy','INCUMBENT_POLICY.json'),('original_orders','local_linear_ORDERS.json')):
  require(inputs[name]==reader.path(LOCAL+'/'+file),'Original local67 input: '+name)
 require(inputs['prices']==reader.path(old.PRICES),'Original prices')
 helper=old.original_helper();policy=reader.json(inputs['policy'])
 require(all(policy.get(k)==v for k,v in helper.POLICY.items()),'Frozen episode policy')
 cache=reader.json(directory/'CACHE.json');require(set(cache)==set(CACHE_FILES),'Six preparation cache artifacts');reader.verify(cache,directory)
 meta=reader.json(directory/'META.json');require(meta==reader.json(inputs['meta']),'Exact original metadata')
 require(len(meta['dates'])==len(meta['symbols'])==122 and meta['train_months']==88 and meta['early_months']==53 and len(meta['etfs'])==17 and all(len(row)==49 for row in meta['symbols']),'Original122x49 shape')
 original=reader.npz(LOCAL+'/DATA.npz');arrays=reader.npz(directory/'DATA.npz');verify_original_arrays(original,arrays)
 require(reader.sha(old.LOADER)==old.LOADER_SHA,'Frozen raw price loader')
 spec=importlib.util.spec_from_file_location('cash_quality146_independent_price_loader',reader.path(old.LOADER));loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
 panel,info=loader.load_panel(meta['etfs']);used=dict(info['source_hashes']);used[str(reader.path(old.LOADER))]=old.LOADER_SHA
 require({str(reader.path(k)):v for k,v in used.items()}==raw,'Exactly the registered raw64 consumed');reader.verify(used)
 prices=reader.npz(inputs['prices']);require(set(prices)=={'dates','symbols','close','opening'},'Original price cache schema')
 parity=old.price_parity(panel,prices);old.compare(reader.json(directory/'PRICE_PARITY.json'),parity,'PRICE_PARITY')
 calendar_path=inputs['sessions'].parent/'CALENDAR_PLAN.json';require(reader.sha(calendar_path)==CALENDAR_SHA,'Pinned source calendar')
 calendar=reader.json(calendar_path);require(calendar['start']=='2015-01-01' and calendar['end']=='2027-12-31','Calendar coverage')
 require(all(pins.get(k)==v for k,v in reader.verify(calendar['bindings']).items()),'Calendar provenance')
 sessions=old.official_sessions(reader,calendar['end']);sessions=[str(d.date()) for d in sessions if str(d.date())>=calendar['start']]
 saved=reader.json(inputs['sessions']);require(saved['plan_sha256']==CALENDAR_SHA and saved['sessions']==sessions,'Complete official session overlay')
 dataset=verify_admission(reader,plan,pins,sessions);expected=features_oracle(meta,dataset)
 verify_audit(reader.json(directory/'FEATURE_AUDIT.json'),expected,dataset_sha=reader.sha(inputs['admitted_dataset']),admission_sha=reader.sha(inputs['source_admission']))
 for key in EXTRA:
  require(arrays[key].shape==(122,49) and arrays[key].dtype==expected['arrays'][key].dtype,'New feature dtype/shape')
  np.testing.assert_array_equal(arrays[key],expected['arrays'][key])
 scores,choices,deltas=score_oracle(arrays,meta,*(expected['arrays'][key] for key in EXTRA[:3]))
 saved_scores=reader.npz(directory/'SCORES.npz');saved_orders=reader.json(directory/'PRETRAIN_ORDERS.json')
 verify_scores_and_orders(saved_scores,saved_orders,meta,scores,choices)
 helper.assert_exact_tree(saved_orders[NAMES[0]],reader.json(inputs['original_orders']))
 require(saved_scores[NAMES[0]].tobytes()==scores[NAMES[0]].tobytes(),'Incumbent score bytes')
 verify_prefixes(arrays,meta,dataset,expected,scores,choices)
 required={str(directory/name):reader.sha(directory/name) for name in (*CACHE_FILES,'PLAN.json','CACHE.json')};allowed=dict(pins,**required)
 for path,digest in list(reader.seen.items()):require(reader.sha(path)==digest and allowed.get(path)==digest,'Unbound consumed artifact: '+path)
 return dict(schema_version=1,stage='prepare',passed=True,plan_sha256=plan_sha,cache_sha256=reader.sha(directory/'CACHE.json'),orders_sha256=reader.sha(directory/'PRETRAIN_ORDERS.json'),
  **{key:True for key in CHECKS},source_hashes=sources,artifact_hashes=dict(reader.seen),
  producer_modules_imported=False,returns_or_nav_computed=False,monthly_label_models_fit=0,new_model_fits=0,
  orders_checked=488,source_cells_checked=5978,stock_cells_checked=3904,available_stock_cells=int(expected['arrays']['common_available'].sum()),
  exact_integer_closed_form_arithmetic=True,source_admission_bound=True,source_union_preservation_verified=True,
  raw_prices_independently_verified=True,calendar_overlay_independently_verified=True,all_signal_prefixes_verified=True,
  max_score_error=max(float(np.max(np.abs(saved_scores[name]-scores[name]))) for name in NAMES),
  limitations=['Raw source assumptions and inherited holds remain; the new gate is conditional research admission.',
   'Integer-scaled bounded ratios and all four ordered score paths are checked without importing the new producer; no strategy returns are computed.'])


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--describe',action='store_true');parser.add_argument('--plan-sha')
 parser.add_argument('--plan',type=Path,default=ROOT/STUDY/'PLAN.json');parser.add_argument('--prepare-complete',action='store_true')
 parser.add_argument('--output',type=Path,default=ROOT/STUDY/'INDEPENDENT_PREPARE.json');args=parser.parse_args()
 if args.describe:
  print(json.dumps(dict(stage='oracles_and_registered_audit_ready',method=METHOD,arms=NAMES,features=EXTRA,checks=CHECKS,actual_checks_run=False,producer_modules_imported=False)));return
 require(args.prepare_complete,'Explicit completed preparation required')
 result=audit(ROOT,args.plan,args.plan_sha)
 require(args.output.resolve()==ROOT/STUDY/'INDEPENDENT_PREPARE.json','Dedicated independent output')
 with args.output.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(passed=True,output=str(args.output),sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))

if __name__=='__main__':main()
