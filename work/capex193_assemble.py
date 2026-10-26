"""Consolidate all44 CAPEX source records only after independent cell review."""
import argparse
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import capex193_admission as gate

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_capex193_source/admitted'
SCOPE='work/capex193_source_scope.json'
SCOPE_SHA='71ccee9c200ae377010fb03a155bc7a6976f7f3f139db70a0a0ebdf49098fe08'
CONCEPT='work/capex193_experiment_concept.json'
CONCEPT_SHA='14e69a47bbf3ea9bbc8d60026e1f5092504d28778f52d622b7560b28bc5c1689'
PARTS={
 'outputs/monthly_capex193_source/msft_aapl/EXTRACTION.json':'288f1af295a25cadbf2eb26437e53513382942fc50db46997164e50b3edebf2d',
 'outputs/monthly_capex193_source/nvda_amd/EXTRACTION.json':'81dbb8535788765e8ec0497d0bc4314b4728a6f5aad6b36525e83143333627ee'}
CALENDAR='outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'
CALENDAR_SHA='c05c15f19bc1f7a3428e456a71199d64f05799ef3c432fa45f7bc485d0ff3cf4'
RUNTIME='work/capex193_pdf_runtime/RUNTIME.json'
RUNTIME_SHA='ad437674d06761ef278da476110c1d3aedf386cd5a0276161304627d2140c98a'
require=gate.require

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def encoded(x):return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
def rows(x):return x.get('records', x.get('events',x.get('event_reviews')))
def path(p):
 p=Path(p);p=p if p.is_absolute() else ROOT/p
 require(p.resolve()==p and p.is_relative_to(ROOT), 'Regular local research path required: '+str(p))
 return p

def validate_reviews(review,visual,original):
 require(review.get('passed') is True,'Independent review failed or incomplete')
 require(visual.get('passed') is True and visual.get('report_count')==10 and visual.get('visual_review_pending') is False,'Ten original PDF pages must be visually reviewed')
 require(visual.get('extraction_sha256')==PARTS['outputs/monthly_capex193_source/msft_aapl/EXTRACTION.json'], 'Visual extraction binding mismatch')
 visual_ids=[e['event_id'] for e in original['events'] if e['symbol']=='AAPL']
 gate.indexed(visual['records'],visual_ids)
 for item in visual['records']:
  require(all(item.get(k) is True for k in ('pure_ppe_row_distinct_from_acquisitions_and_intangibles','legible_unclipped_annual_columns','extracted_value_matches_visual','producer_visual_review_completed')), 'Incomplete visual page check')
 require(review['producer_refs']==PARTS, 'Independent producer references differ')
 require(review['scope_sha256']==SCOPE_SHA and review['runtime']['manifest_sha256']==RUNTIME_SHA, 'Independent scope/runtime binding differs')
 require(review['calendar_sha256']==CALENDAR_SHA and review['runtime']['historical_runtime_currently_passes'] is False, 'Review calendar or historical runtime statement changed')
 require(review.get('source_only') is True and all(review.get(k)==0 for k in ('ratios_computed','factors_computed','returns_computed')), 'Review must be source only')
 require(all(visual.get(k)==0 for k in ('financial_ratios','factors','returns','strategy_admissions','fact_admissions')), 'Visual review must not grant strategy admission')
 require(len(visual_ids)==10, 'Exact ten original AAPL visual reports required')
 independent_visual=gate.indexed(review['independent_visual_reviews'],visual_ids)
 for item in visual['records']:
  reviewed=independent_visual[item['event_id']]
  require(reviewed['source_sha256']==item['source']['sha256'] and reviewed['render_path']==item['render']['path'] and reviewed['render_sha256']==item['render']['sha256'], 'Independent visual source/render mismatch')
  require(all(reviewed.get(k) is True for k in ('current_annual_column_first','units_millions','pure_cash_ppe_separate_from_intangibles','cfo_matches_source','independent_visual_review')), 'Independent visual check incomplete')

def main(review_path, review_sha, visual_path, visual_sha):
 hashes={}
 def read(p,h):
  f=path(p); require(digest(f)==h, 'Input SHA mismatch: '+str(p));hashes[str(f.relative_to(ROOT))]=h
  return json.loads(f.read_text())
 def bind(p,h):
  f=path(p);require(digest(f)==h,'Closure SHA mismatch: '+str(p));hashes[str(f.relative_to(ROOT))]=h
 scope=read(SCOPE,SCOPE_SHA);concept=read(CONCEPT,CONCEPT_SHA)
 require(concept['registered'] is False and all(concept[k]==0 for k in ('actual_new_factors','actual_new_orders','actual_new_returns')), 'Source gate must precede actual experiment')
 old={p:read(p,h) for p,h in scope['input_sha256'].items()}
 original=old['outputs/monthly_cash_quality_source_admission_v1/DATASET.json']
 oldreceipt=old['outputs/monthly_cash_quality_source_admission_v1/ADMISSION.json']
 require(oldreceipt['admitted_events']==31 and oldreceipt['held_events']==13, 'Original31/13 required')
 sessions=read(CALENDAR,CALENDAR_SHA)['sessions']
 runtime=read(RUNTIME,RUNTIME_SHA)
 for p,h in runtime['pdf_runtime_hashes'].items():bind(p,h)
 acquisition=read(runtime['acquisition']['path'],runtime['acquisition']['sha256'])
 require(acquisition['package_executed'] is False and acquisition['RECORD_verified_members']==70, 'Independent acquisition record required')
 for target in scope['prospective_targets']:
  for ref in (target['source'],target['source']['receipt']):bind(ref['path'],ref['sha256'])
 parts=[read(p,h) for p,h in PARTS.items()]
 review=read(review_path,review_sha);visual=read(visual_path,visual_sha)
 validate_reviews(review,visual,original)
 bind('work/capex193_msft_aapl_visual_review.py',visual['review_code_sha256'])
 for obj in parts+[review,visual]:
  for key in ('input_hashes','source_hashes','source_code_hashes','input_sha256','code_sha256','code_hashes','pdf_runtime_hashes','source_pins','producer_code_sha256','producer_refs','render_sha256','image_hashes','test_ref','independent_expected_ref','producer_visual_review_ref'):
   for p,h in obj.get(key,{}).items():bind(p,h)
 required_review_refs={**PARTS,SCOPE:SCOPE_SHA,RUNTIME:RUNTIME_SHA}
 reviewed={**review['producer_refs'],SCOPE:review['scope_sha256'],RUNTIME:review['runtime']['manifest_sha256']}
 require(review['calendar_sha256']==CALENDAR_SHA and review['runtime']['historical_runtime_currently_passes'] is False,'Review calendar or historical runtime statement changed')
 for p,h in required_review_refs.items():
  require(reviewed.get(p,reviewed.get(str(ROOT/p)))==h,'Independent review missing producer/scope/runtime binding: '+p)
 ids=[e['event_id'] for e in original['events']]
 require(len(ids)==44 and sum(e['cash_quality_admission']['admitted'] for e in original['events'])==31,'Original event census changed')
 target_map=gate.indexed(scope['prospective_targets'],ids)
 producer_map=gate.indexed([e for obj in parts for e in rows(obj)],ids)
 reviewer_map=gate.indexed(rows(review),ids)
 result=[gate.check_record(e,target_map[e['event_id']],producer_map[e['event_id']],reviewer_map[e['event_id']],sessions) for e in original['events']]
 for p in ('work/capex193_admission.py','work/test_capex193_admission.py','work/capex193_assemble.py','work/test_capex193_assemble.py'):
  bind(p,digest(ROOT/p))
 policy={'schema_version':1,'method':'current_annual_cash_after_ppe_v1','definition':gate.DEFINITION,'source_scope_sha256':SCOPE_SHA,'concept_sha256':CONCEPT_SHA,
  'created_utc':datetime.now(timezone.utc).isoformat(),'runtime':sys.version,
  'current_report_cash_ppe_only':True,'use_comparison_flows':False,'preserve_all_original_holds':True,
  'clock':'first pinned exchange session strictly after report release date','annual_days':[330,400],'max_age_days':550,
  'same_report_literal_currency_assumption':True,'conditional_research_only':True,
  'financial_ratios_permitted':False,'factor_computation_permitted':False,'return_computation_permitted':False,
  'scope_limitations':scope['source_limitations'],
  'old_extraction_runtime_status':'Historical receipt preserved; old shared PDFium current hash fails. New193 source review uses independently pinned official same-version wheel and original-hash pypdf copies. No old-runtime verification pass is claimed.',
  'input_hashes':hashes}
 policy_sha=hashlib.sha256(encoded(policy)).hexdigest()
 dataset={'schema_version':1,'method':'current_annual_cash_after_ppe_v1','events':result,'policy_sha256':policy_sha,'conditional_research_only':True}
 dataset_sha=hashlib.sha256(encoded(dataset)).hexdigest()
 admitted=[e['event_id'] for e in result if e['cash_ppe_admission']['admitted']]
 receipt={'schema_version':1,'stage':'conditional_cash_ppe_source_admission','passed':True,'events':44,'admitted_events':len(admitted),'held_events':44-len(admitted),
  'admitted_event_ids':admitted,'held_event_ids':[e['event_id'] for e in result if not e['cash_ppe_admission']['admitted']],
  'inherited_admitted_events':31,'inherited_held_events':13,'all_original_event_values_preserved':True,
  'data_sha256':dataset_sha,'policy_sha256':policy_sha,'independent_review':{'path':review_path,'sha256':review_sha},
  'first_public_version_verified':False,'complete_correction_history_verified':False,'certified_daily_security_master':False,
  'financial_ratios':0,'factor_calculations':0,'return_calculations':0,'observer_changes':0}
 require(not OUT.exists(),'Write-once source admission already exists')
 # Recheck every new closure member immediately before publishing.
 for p,h in hashes.items():require(digest(ROOT/p)==h,'Source changed during admission: '+p)
 OUT.mkdir(parents=True)
 for name,obj in [('POLICY.json',policy),('DATASET.json',dataset),('ADMISSION.json',receipt)]:
  with (OUT/name).open('xb') as f:f.write(encoded(obj))
 print(json.dumps({'passed':True,'events':44,'admitted':len(admitted),'held':44-len(admitted),'path':str(OUT.relative_to(ROOT)),'dataset_sha256':dataset_sha}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--review',required=True);p.add_argument('--review-sha',required=True);p.add_argument('--visual',required=True);p.add_argument('--visual-sha',required=True);a=p.parse_args()
 main(a.review,a.review_sha,a.visual,a.visual_sha)
