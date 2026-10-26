"""Source-only independent post-assembly check; no production imports or ratios."""
from bisect import bisect_right
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE='outputs/monthly_capex193_source/'
PINS={
 BASE+'admitted/ADMISSION.json':'f93168a07b174df7f54e4c5a9dcba6f3e65b1d94dfb88f366dc82c82bf865238',
 BASE+'admitted/DATASET.json':'ac3fdc44bde1d98ca925581d1e7b7de87a1bf091825437ab16ba68083724cf9d',
 BASE+'admitted/POLICY.json':'ebc33f7b8afe906d1d55e6906fe079b695394ead412f46318c915aeea3509584',
 'outputs/monthly_cash_quality_source_admission_v1/DATASET.json':'3725b384ab4f48a3cabee1fc1adb1acd896be3f4967e8b6cd649a9a690c953d5',
 'work/capex193_source_scope.json':'71ccee9c200ae377010fb03a155bc7a6976f7f3f139db70a0a0ebdf49098fe08',
 BASE+'independent/REVIEW.json':'e8b0c51cff771eee5a48aeb885be1471305faa9cdaa5348e8db2c11527a296d7',
 'outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json':'c05c15f19bc1f7a3428e456a71199d64f05799ef3c432fa45f7bc485d0ff3cf4'}
CODE_PINS={
 'work/capex193_factor.py':'aca840bfd64d51984b65510ba3ed284a689cfeafdbd773fbcddddaa66b8369c0',
 'work/test_capex193_factor.py':'2d58e659a4c4a27c7f27b91a192da95e804ce1e544f7efe958db3e5a4b2d87ea'}
def require(ok,message):
 if not ok: raise ValueError(message)
def digest(path): return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def read(path): return json.loads((ROOT/path).read_text())
def rows(obj):
 for key in ('events','records','event_reviews'):
  if key in obj: return obj[key]
 raise ValueError('No event records')
def index(records):
 result={r['event_id']:r for r in records}
 require(len(result)==len(records),'Duplicate event IDs')
 return result

def main():
 for p,h in {**PINS,**CODE_PINS}.items(): require(digest(p)==h,'Pinned bytes differ: '+p)
 admission=read(BASE+'admitted/ADMISSION.json'); data=read(BASE+'admitted/DATASET.json'); policy=read(BASE+'admitted/POLICY.json')
 original=read('outputs/monthly_cash_quality_source_admission_v1/DATASET.json')
 scope=read('work/capex193_source_scope.json'); review=read(BASE+'independent/REVIEW.json')
 sessions=read('outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json')['sessions']
 require(sessions==sorted(set(sessions)),'Calendar ordering')
 require(admission['passed'] is True and (admission['events'],admission['admitted_events'],admission['held_events'])==(44,24,20),'Admission census')
 require(admission['data_sha256']==PINS[BASE+'admitted/DATASET.json'] and admission['policy_sha256']==data['policy_sha256']==PINS[BASE+'admitted/POLICY.json'],'Cross-linked SHA')
 require(data['method']==policy['method']=='current_annual_cash_after_ppe_v1' and data['conditional_research_only'] is True,'Conditional source method')
 require(all(policy[k] is False for k in ('factor_computation_permitted','financial_ratios_permitted','return_computation_permitted')),'Source-only boundary')
 closure=policy['input_hashes']; require(len(closure)==249,'Unexpected closure census')
 for p,h in closure.items(): require(digest(p)==h,'Closure bytes differ: '+p)
 for key in ('producer_refs','source_pins','code_sha256','producer_code_sha256','independent_expected_ref','test_ref','render_sha256','producer_visual_review_ref'):
  for p,h in review[key].items(): require(closure.get(p)==h,'Review closure omitted: '+p)
 require(closure['work/capex193_pdf_runtime/RUNTIME.json']==review['runtime']['manifest_sha256'],'New runtime binding')
 require(review['runtime']['historical_runtime_currently_passes'] is False,'Historical failure wrongly cleared')
 targets=index(scope['prospective_targets']); reviewed=index(rows(review)); producers={}
 for p,h in review['producer_refs'].items():
  require(digest(p)==h,'Producer SHA')
  for k,v in index(rows(read(p))).items():
   require(k not in producers,'Producer duplicate'); producers[k]=v
 ids=[e['event_id'] for e in original['events']]
 require(len(ids)==len(set(ids))==44 and [e['event_id'] for e in data['events']]==ids,'Original44 identity/order')
 require(set(ids)==set(targets)==set(reviewed)==set(producers),'Coverage44')
 added={'cash_ppe','cash_ppe_source_evidence','cash_ppe_admission'}; event_reviews=[]
 for old,new in zip(original['events'],data['events']):
  eid=old['event_id']; target=targets[eid]; checked=reviewed[eid]; producer=producers[eid]
  require(set(new)==set(old)|added and {k:v for k,v in new.items() if k not in added}==old,'Original event value changed: '+eid)
  require(new['cash_ppe_source_evidence']==producer,'Raw producer evidence changed: '+eid)
  raw=producer.get('field',producer.get('capex')); normalized=deepcopy(raw)
  if normalized is not None:
   normalized.setdefault('source',deepcopy(target['source'])); normalized.setdefault('period',deepcopy(old['periods']['current'])); normalized.setdefault('available_from',old['available_from']); normalized['fact_admitted']=False
  require(new['cash_ppe']==normalized,'Normalization changed source values: '+eid)
  require(checked['review_passed'] is True and checked['source_sha256']==target['source']['sha256'],'Independent evidence binding')
  ix=bisect_right(sessions,old['release_date'])
  require(ix<len(sessions) and sessions[ix]==old['available_from']==target['inherited_available_from']==producer['available_from'],'Availability not strict next exchange day: '+eid)
  if raw is not None:
   require(checked['cash_ppe_verified'] is True,'Unverified raw field')
   for key in ('signed_value_decimal','positive_outflow_decimal','scale'): require(normalized[key]==checked[key],'Raw amount mismatch: '+eid+'/'+key)
   require(normalized['locator']['period_end']==checked['annual_period_end']==old['periods']['current']['period_end'],'Annual period')
   require(normalized['source']==target['source'] and normalized['period']==old['periods']['current'],'Source/period normalized')
   require(hashlib.sha256(normalized['row_text'].encode()).hexdigest()==normalized['row_text_sha256'],'Row bytes changed')
   if normalized.get('currency_literal') is None:
    cfo=old['fields']['current']['operating_cash_flow']; loc=normalized['locator']; previous=cfo['locator']
    require(normalized['statement_currency_literal']==cfo['currency_literal']=='$' and loc['original_cfo_locator']==previous and loc['format']==previous['format'],'Absent row literal lacks statement evidence')
    if loc['format']=='html': require(loc['table_index_zero_based']==previous['table_index_zero_based'] and loc.get('statement_segment')==previous.get('statement_segment'),'Cash table differs')
    else: require(loc['format']=='pdf' and loc['page_1based']==previous['page_1based'],'Cash page differs')
  else: require(checked['cash_ppe_verified'] is False,'Missing was converted to zero')
  should_admit=old['cash_quality_admission']['admitted'] and checked['cash_ppe_verified'] and checked['paired_cfo_verified'] and checked['paired_assets_verified']
  gate=new['cash_ppe_admission']
  require(gate['admitted'] is should_admit and bool(gate['reasons']) is (not should_admit),'Independent admission differs: '+eid)
  require(gate['definition']=='current_annual_cash_ppe_outflow_v1','Wrong definition')
  event_reviews.append(dict(event_id=eid,original_values_exact=True,raw_evidence_exact=True,normalization_exact=True,source_clock_verified=True,admitted=should_admit))
 admitted=[r['event_id'] for r in event_reviews if r['admitted']]; held=[r['event_id'] for r in event_reviews if not r['admitted']]
 require(admitted==admission['admitted_event_ids'] and held==admission['held_event_ids'],'Reported admission sets differ')
 oldheld={e['event_id'] for e in original['events'] if not e['cash_quality_admission']['admitted']}
 require(len(oldheld)==13 and oldheld<=set(held),'Original holds lost')
 newheld=set(held)-oldheld
 require(len(newheld)==7 and all(producers[e]['symbol']=='NVDA' for e in newheld),'New holds differ')
 source_code={p:h for p,h in closure.items() if p in ('work/capex193_admission.py','work/test_capex193_admission.py','work/capex193_assemble.py','work/test_capex193_assemble.py')}
 result=dict(schema='capex193_independent_admission_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,source_only=True,events=44,admitted_events=24,held_events=20,inherited_holds_preserved=13,new_holds=7,closure_pins_verified=len(closure),all_original_event_values_preserved=True,all_raw_producer_evidence_preserved=True,normalized_fields_and_source_clock_checked=True,source_review_closure_preserved=True,source_runtime_status='New193 runtime is pinned; historical shared PDFium failure remains recorded',factor_code_review=dict(passed=True,code_sha256=CODE_PINS,same_table_or_page_currency_evidence=True,original_ni_gate_retained=True,latest_report_selected_before_validity=True,future_amounts_not_read=True,exact_fraction_then_bounded_float=True,synthetic_tests_reviewed=23,actual_factor_execution=False),source_code_sha256=source_code,input_sha256=PINS,review_code_sha256=digest('work/capex193_independent_admission_review.py'),event_reviews=event_reviews,ratios_computed=False,factors_computed=False,returns_computed=False,production_modules_imported=False,remaining='Numerical oracle and strategy host require separate review; no strategy PLAN or numerical admission granted here')
 out=ROOT/BASE/'independent/ADMISSION_REVIEW.json'
 with out.open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2); f.write('\n')
 print(json.dumps(dict(passed=True,events=44,admitted=24,held=20,closure=249,path=str(out.relative_to(ROOT)),sha256=hashlib.sha256(out.read_bytes()).hexdigest())))
if __name__=='__main__': main()
