"""Write-once metadata PLAN and source-only annual extraction for fixed batch5.

All actual financial cells remain untouched until the explicit extract phase.
"""
import argparse
from collections import Counter
import copy
from datetime import date,timedelta
from decimal import Decimal
import io
import json
import re
import unittest

import after149_source_batch as shared
import after153_annual_extract as parser

ROOT=shared.ROOT
OUT=ROOT/'outputs/monthly_financial_coverage_batch5_v1'
CONTEXT='work/after153_next_source_context.json'
CONTEXT_SHA='1bd968ca401a313af2ee6c1e92328ae8b270e7c78e607b14edb61060ecc6f241'
FEASIBILITY='work/after153_next_source_feasibility.json'
FEASIBILITY_SHA='4e7c8025fa8f05cb7e48d874e34bc583e2968254fce6479723fd237a1cc1353a'
POLICY='work/after153_batch5_admission_policy.json'
CENSUS='outputs/monthly_financial_coverage_census_v1/CENSUS.json'
CONCEPT='work/after149_coverage_extension_concept.json'
SESSIONS='outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'
PRIOR='outputs/monthly_financial_coverage_admission_batch4_v1/'
BASE_INPUTS=(CONTEXT,FEASIBILITY,CENSUS,CONCEPT,SESSIONS)+tuple(PRIOR+n for n in ('DATASET.json','ADMISSION.json','MANIFEST.json'))
READY='standard_HTML_annual_header_context_located_not_admitted'
FIELDS=parser.FIELDS
FULL=[('PLTR',y) for y in range(2020,2026)]+[('PYPL',y) for y in range(2015,2026)]+[('TMO',y) for y in range(2012,2023)]
SELECTED=[('PLTR',y) for y in range(2020,2026)]+[('PYPL',y) for y in range(2015,2020)]+[('TMO',y) for y in range(2012,2022)]
check=shared.check


def records(context):
 events=context['events']
 check(len(events)==len({e['event_id'] for e in events})==28,'Complete unique28 catalog required')
 check([(e['symbol'],e['fiscal_year']) for e in events]==FULL,'Changed full catalog order')
 selected=[e for e in events if e['context_status']==READY]
 check([(e['symbol'],e['fiscal_year']) for e in selected]==SELECTED,'Changed fixed21 source scope')
 check(all(e['source_kind']=='HTML_original' and e['original_report_bytes_verified'] is True for e in selected),'Original HTML required')
 return selected


def catalog_metadata(context):
 keys=('event_id','symbol','fiscal_year','publication_date','available_session','available_session_verified','official_actual_period_end','source','source_kind','original_report_bytes_verified','context_status','context_holds')
 return [{k:copy.deepcopy(e[k]) for k in keys} for e in context['events']]


def original_catalog(census,context):
 rows=[e for e in census['annual_documents'] if e['symbol'] in parser.PROFILES]
 check(len(rows)==28,'Original census must contain28 entries')
 for raw,ctx in zip(rows,context['events']):
  check(all(ctx.get(k)==v for k,v in raw.items() if k!='source_kind'),'Original census metadata changed')
  check(ctx.get('census_source_kind')==raw['source_kind'],'Original source kind changed')
 return copy.deepcopy(rows)


def make_policy(context,inputs,created_utc=None):
 records(context)
 check(set(inputs)==set(BASE_INPUTS) and all(re.fullmatch('[0-9a-f]{64}',h) for h in inputs.values()),'Exact policy inputs required')
 check(inputs[CONTEXT]==CONTEXT_SHA and inputs[FEASIBILITY]==FEASIBILITY_SHA,'Fixed metadata digests required')
 return dict(schema_version=1,policy_id='pltr_pypl_tmo_annual_ni_same_report_assets_batch5_v1',
  stage='prospective_source_policy_not_strategy_registration',created_utc=created_utc or shared.now(),definition='current_annual_net_income_average_assets_v1',
  purpose='Extract current annual consolidated GAAP total NI and same-report annual asset endpoints; do not calculate ratios.',
  source_scope=dict(symbols=list(parser.PROFILES),catalog_events=28,extracted_events=21,unprocessed_catalog_events=7,selection=[list(x) for x in SELECTED],
   reason='Remaining three issuers in the prior fixed alphabetic source preparation list; all structurally identified annual pairs, regardless of returns.',
   no_automatic_admission=True),fixed_catalog_metadata=catalog_metadata(context),required_fields=list(FIELDS),
  net_income_scope=dict(required='Total consolidated net income or loss including discontinued operations, before parent/noncontrolling attribution.',
   prohibited=['quarter or YTD substitution','EPS','non-GAAP or pro-forma','attributable-to-parent NI','CFO as NI','annualizing or summing quarters'],
   negative_zero='Preserve both signed loss and literal zero.'),
  periods=dict(current='Explicit full current annual NI end and current total-assets end must agree.',
   assets='Same current report closing and previous annual closing Total assets.',allowed_days=[365,366],required_year_end='12-31',
   boundary_assumptions={s:dict(assumption_id=s+'_consecutive_annual_boundaries_v1',formula='prior annual assets date + one calendar day',
    not_a_quoted_fact=True,requires_independent_header_verification=True) for s in parser.PROFILES},
   rejection='No367-day expansion,53-week substitution,stub period or guessed calendar start; preserve explicit gaps.'),
  numeric=dict(scale='Verify each printed field/table scale; PLTR thousands, PYPL/TMO millions only if actually printed.',
   currency='Common report dollar symbol, currency_iso null; not ISO certification.',
   literal='Preserve signed exact decimal, original text and physical coordinates; null when ambiguous; never zero-fill.'),
  known_time=dict(available_from='First pinned exchange session strictly after source publication, not the fiscal end.',
   early_calendar='Keep the two early TMO clock holds; unverified clocks may never become known through weekday inference.',
   latest='Latest known catalog event chosen before validity; held, unprocessed or equal-date ambiguity blocks fallback.',
   age_days=550,age_from='current annual period end',first_public_version_verified=False),
  accounting=dict(policy='Field-specific independent evidence is required for current NI plus both assets; keywords alone neither reject nor clear.',
   pending=['PYPL FY2015 combined/carve-out basis after eBay separation','TMO FY2018 pension restatement and opening assets scope'],
   field_choices=['PLTR2022+ total NI before attribution','TMO total NI including discontinued operations; total not parent-attributable in2021'],
   all_context_holds_retained=True),
  preservation=dict(original44_and_batches1_to4='Referenced unchanged; no migration of old admissions.',all28_original_catalog_entries=True,
   stock_month_axis=3904,no_observer_changes=True,no_new_strategy_trials=True,no_ratio_or_score_or_return=True,
   no_first_public_time_or_currency_ISO_or_security_master_certification=True),input_hashes=copy.deepcopy(inputs))


def validate_policy(policy,context):
 check(policy==make_policy(context,policy['input_hashes'],policy['created_utc']),'Changed fixed source policy')


def validate_record(value,context):
 for key in ('symbol','event_id','fiscal_year','source','context_holds','publication_date'):
  check(value.get(key)==context[key],'Changed event '+key)
 for key,default in (('accounting_review_flags',[]),('accounting_context_numbers_redacted',[])):
  ck='accounting_terms' if key=='accounting_review_flags' else key
  check(value.get(key)==context.get(ck,default),'Changed accounting context')
 for flag in ('fact_admitted','model_admitted','strategy_admitted','first_public_version_verified','first_public_time_verified','complete_correction_history_verified','certified_daily_security_master'):
  check(value.get(flag) is False,'Premature certification '+flag)
 check(type(value.get('financial_calculations')) is int and value['financial_calculations']==0,'Forbidden calculations')
 check(value.get('available_from')==context['available_session'],'Changed available time')
 check(set(value['fields'])==set(FIELDS),'Changed field names')
 expected=parser.periods(context)
 check(value.get('periods')==expected,'Changed annual period/assumption')
 gaps=value.get('gaps',[])
 check(all(isinstance(g,dict) and g.get('field') in FIELDS and isinstance(g.get('reason'),str) and g['reason'] for g in gaps),'Invalid explicit gaps')
 check(len(gaps)==len({g['field'] for g in gaps}),'Duplicate gap')
 for name,f in value['fields'].items():
  gap=any(g['field']==name for g in gaps)
  if f is None:check(gap,'Null without reason');continue
  check(not gap,'Nonnull with contradictory gap')
  check(f.get('source')==context['source'] and f.get('available_from')==context['available_session'] and f.get('fact_admitted') is False,'Changed field provenance')
  check(type(f.get('value_decimal')) is str and type(f.get('scale')) is str,'Exact decimal strings required')
  amount,scale=Decimal(f['value_decimal']),Decimal(f['scale'])
  check(amount.is_finite() and scale.is_finite() and scale>0 and f['scale']==context['fields'][name]['scale'],'Invalid amount/scale')
  check(f.get('currency_literal')=='$' and 'currency_iso' in f and f['currency_iso'] is None,'Changed currency assumption')
  check(isinstance(f.get('displayed_amount'),str) and f['displayed_amount'] and bool(f.get('locator')),'Missing literal/locator')
  end=expected['comparison']['period_end'] if name=='prior_total_assets' else expected['current']['period_end']
  check(f.get('as_of')==end and f.get('period')==(expected['current'] if name=='annual_net_income' else None),'Changed field period')
 check(value.get('complete_required_fields') is (not gaps),'Inconsistent field completeness')


def synthetic_tests():
 answer=[]
 for module in ('test_after153_source_batch','test_after153_next_source_context','test_after153_annual_extract'):
  output=io.StringIO();r=unittest.TextTestRunner(stream=output,verbosity=0).run(unittest.defaultTestLoader.loadTestsFromName(module))
  check(r.wasSuccessful() and r.testsRun>0,module+'\n'+output.getvalue())
  answer.append(dict(module=module,tests=r.testsRun,passed=True,output=output.getvalue()))
 return answer


def pins(mapping,target):
 for raw,h in mapping.items():
  name=shared.relative(raw)
  check(re.fullmatch('[0-9a-f]{64}',h) and shared.sha(name)==h,'Changed dependency '+name)
  check(name not in target or target[name]==h,'Conflicting pin '+name);target[name]=h


def code_files():
 return ['work/after153_source_batch.py','work/test_after153_source_batch.py']+list(parser.source_files())


def freeze():
 check(not (OUT/'PLAN.json').exists(),'PLAN already exists')
 check(shared.sha(CONTEXT)==CONTEXT_SHA and shared.sha(FEASIBILITY)==FEASIBILITY_SHA,'Changed fixed metadata')
 context=shared.load(CONTEXT);policy=shared.load(POLICY);validate_policy(policy,context)
 bound={POLICY:shared.sha(POLICY)};pins(context['input_hashes'],bound);pins(policy['input_hashes'],bound)
 original=original_catalog(shared.load(CENSUS),context)
 for e in context['events']:
  s=e['source'];pins({s['path']:s['sha256']},bound)
  check((s['receipt_path'] is None)==(s['receipt_sha256'] is None),'Partial receipt')
  if s['receipt_path'] is not None:pins({s['receipt_path']:s['receipt_sha256']},bound)
  else:check(e['source_kind']!='HTML_original','Missing original receipt')
 code=shared.code_closure(code_files());tests=synthetic_tests()
 check(code==shared.code_closure(code_files()),'Code changed during synthetic validation');pins(bound,{})
 plan=dict(schema_version=1,created_utc=shared.now(),stage='frozen_source_only_plan',runtime=shared.runtime(),purpose=policy['purpose'],
  context=CONTEXT,policy=POLICY,parser='after153_annual_extract',fields=list(FIELDS),catalog_events=28,
  catalog_ids=[e['event_id'] for e in context['events']],catalog_metadata=catalog_metadata(context),original_catalog_metadata=original,
  records=[dict(event_id=e['event_id'],symbol=e['symbol'],fiscal_year=e['fiscal_year'],source_path=e['source']['path'],source_sha256=e['source']['sha256']) for e in records(context)],
  inputs=bound,code=code,synthetic_tests=tests,no_admission_by_extraction=True,new_strategy_trials=0,preserve_all_prior_source_holds=True,
  preserve_unprocessed_catalog_events=True,no_network=True,no_observer_changes=True,no_scores_or_returns=True)
 shared.write_new(OUT/'PLAN.json',plan)
 return dict(path=str((OUT/'PLAN.json').relative_to(ROOT)),sha256=shared.sha(OUT/'PLAN.json'),records=21,catalog_events=28)


def run(plan_sha):
 check(shared.sha(OUT/'PLAN.json')==plan_sha,'Exact PLAN SHA required');plan=shared.load(OUT/'PLAN.json')
 check(plan['stage']=='frozen_source_only_plan' and plan['runtime']==shared.runtime(),'Changed PLAN stage/runtime')
 check(plan['parser']=='after153_annual_extract' and plan['fields']==list(FIELDS) and plan['catalog_events']==28,'Changed extraction protocol')
 check(plan['code']==shared.code_closure(code_files()),'Changed extraction code closure')
 check(not (OUT/'EXTRACTIONS.json').exists() and not (OUT/'EXTRACTION_ERRORS.json').exists(),'Completed or failed batch immutable')
 bound={};pins(plan['inputs'],bound);pins(plan['code'],bound)
 context=shared.load(CONTEXT);validate_policy(shared.load(POLICY),context)
 check(plan['catalog_metadata']==catalog_metadata(context) and plan['original_catalog_metadata']==original_catalog(shared.load(CENSUS),context),'Changed catalog')
 chosen=records(context)
 check(plan['records']==[dict(event_id=e['event_id'],symbol=e['symbol'],fiscal_year=e['fiscal_year'],source_path=e['source']['path'],source_sha256=e['source']['sha256']) for e in chosen],'Changed fixed extraction scope')
 values=[];errors=[]
 for c in chosen:
  try:
   value=parser.extract_document(shared.path(c['source']['path']).read_bytes(),c);validate_record(value,c);values.append(value)
  except Exception as exc:errors.append(dict(event_id=c['event_id'],error_type=type(exc).__name__,error=str(exc)))
 if errors:
  shared.write_new(OUT/'EXTRACTION_ERRORS.json',dict(plan_sha256=plan_sha,created_utc=shared.now(),errors=errors,successful_events_not_published=len(values),new_financial_admissions=0,new_strategy_trials=0))
  raise ValueError('Errors preserved; no partial batch published')
 check(len(values)==21,'Incomplete batch');pins(bound,{})
 result=dict(schema_version=1,created_utc=shared.now(),plan_sha256=plan_sha,status='source_values_extracted_pending_independent_review',
  catalog_ref=dict(path=CONTEXT,sha256=CONTEXT_SHA),catalog_ids=plan['catalog_ids'],catalog_metadata=plan['catalog_metadata'],original_catalog_metadata=plan['original_catalog_metadata'],
  counts=dict(events=21,by_symbol=dict(Counter(e['symbol'] for e in values)),nonnull_fields=sum(e['fields'][f] is not None for e in values for f in FIELDS),unprocessed_catalog_events=7),
  events=values,new_financial_admissions=0,new_strategy_trials=0,ratios_computed=0,returns_computed=0,
  limitations=['Raw extraction is neither independent audit nor admission.','All28 original catalog entries and context holds remain.','No first-public clock/version,ISO currency or security-master certification.'])
 shared.write_new(OUT/'EXTRACTIONS.json',result)
 return dict(path=str((OUT/'EXTRACTIONS.json').relative_to(ROOT)),sha256=shared.sha(OUT/'EXTRACTIONS.json'),counts=result['counts'])


def main():
 cli=argparse.ArgumentParser();cli.add_argument('--phase',required=True,choices=('policy','tests','freeze','extract'));cli.add_argument('--plan-sha');args=cli.parse_args()
 if args.phase=='policy':
  policy=make_policy(shared.load(CONTEXT),{p:shared.sha(p) for p in BASE_INPUTS});shared.write_new(POLICY,policy);result=dict(path=POLICY,sha256=shared.sha(POLICY))
 elif args.phase=='tests':result=synthetic_tests()
 elif args.phase=='freeze':result=freeze()
 else:check(args.plan_sha is not None,'Exact PLAN SHA required');result=run(args.plan_sha)
 print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
