"""Hash-bound source-only union of the fixed 44 cashflow events.

No parser, fallback, data admission, ratio, factor, model or return calculation.
Draft/freeze inspect contracts only; build compares already extracted values.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import cashflow_source_preflight_v1 as base

ROOT=Path(__file__).resolve().parents[1]
SELF='work/cashflow_source_union_v1.py'
TEST='work/test_cashflow_source_union_v1.py'
OUT='outputs/monthly_cashflow_source_union_v1'
DIRECTORIES={'baseline':'outputs/monthly_cashflow_source_preflight_v1',
 'pdf':'outputs/monthly_cashflow_pdf_repair_v2','html':'outputs/monthly_cashflow_html_repair_v2'}
STAGES={'baseline':'frozen_cashflow_source_preflight_plan','pdf':'frozen_cashflow_pdf_repair_plan','html':'frozen_HTML_repair_plan'}
BASELINE_PINS={
 DIRECTORIES['baseline']+'/PLAN.json':'4161a752cfac3039ea733b000cab647b8c77c48562ef6f497d109aefc7be1472',
 DIRECTORIES['baseline']+'/RAW_FACTS.json':'4f6d6e694195a4986f911f7ba1defc24efc515b49aa510b5c4be61df78c45c87',
 DIRECTORIES['baseline']+'/AUDIT.json':'43e908b15cf4ceb5fdfce700d74c6ff3788b906cff8f485c8c5e55026161c143',
 DIRECTORIES['baseline']+'/MANIFEST.json':'ee636a289dab86fe4758b60f0073bbad5be2ecd327f2cce552235719ec10a068',
 base.SELF:'c19b3b76bc2cabed6b4c1e3873be1a76e88e7675178debdc7b063e44afe5fbc9',
 'work/same_report144_admission.py':'afb62d9286327d9849d9c42ca579c4566cbdc3bfdf7bcdc2274bfd8b30fd41c1'}
FIELDS=('operating_cash_flow','net_income','total_assets')
SIDES=('current','comparison')
REQUIRED=(('current','operating_cash_flow'),('current','net_income'),('current','total_assets'),('comparison','total_assets'))
ADMISSIONS=('fact_admitted','model_admitted','strategy_admitted')
CERTIFICATIONS=('first_public_version_verified','complete_correction_history_verified','certified_daily_security_master')
ZERO_COUNTS=('fact_admissions','model_admissions','strategy_admissions','financial_calculations','ratios_computed','signs_computed','ranks_computed','returns_computed','model_fits')
ADDED_EVENT_KEYS={'union_provenance','required_field_completeness'}
require=base.require


def code_hashes(root):return {p:base.sha(root,p) for p in (SELF,TEST)}


def exact(obj,keys,label):base.safe.exact(obj,keys,label)


def indexed(rows,label):
 require(type(rows)is list,label+' must be a list')
 result={}
 for row in rows:
  require(type(row)is dict and type(row.get('event_id'))is str,label+' event ID required')
  require(row['event_id'] not in result,label+' duplicate event ID')
  result[row['event_id']]=row
 return result


def counted(data):
 return sum(e['fields'][s][f] is not None for e in data['events'] for s in SIDES for f in FIELDS)


def source_only(data):
 for key in CERTIFICATIONS:
  if key in data:require(data[key]is False,'Source union cannot certify historical identity/version coverage')
 for key in ZERO_COUNTS:
  require(type(data.get(key))is int and data[key]==0,'Source-only counter required: '+key)
 for event in data['events']:
  for key in CERTIFICATIONS:
   if key in event:require(event[key]is False,'Source event cannot certify historical identity/version coverage')
  for key in ADMISSIONS:require(event.get(key)is False,'Producer event admission forbidden')
  exact(event['fields'],SIDES,'field sides')
  for side in SIDES:
   exact(event['fields'][side],FIELDS,'financial positions')
   for cell in event['fields'][side].values():
    if cell is None:continue
    require(type(cell)is dict and cell.get('fact_admitted')is False,'Raw cell admission forbidden')
    for key in ADMISSIONS:
     if key in cell:require(cell[key]is False,'Raw cell admission forbidden')
  for key in ('gaps','metadata_gaps','accounting_review_flags'):
   require(type(event.get(key))is list,'Explicit producer '+key+' required')
 count=counted(data);counts=data['counts'];n=len(data['events'])
 require({k:counts.get(k) for k in ('events','requested_cells','extracted_cells','gap_cells')}==dict(events=n,requested_cells=n*6,extracted_cells=count,gap_cells=n*6-count),'Producer coverage counts inconsistent')


def binding_contract(value):
 exact(value,('schema_version','stage','source_union_authorized','admission_authorized','producers','accounting_review'),'review bindings')
 require(value['schema_version']==1 and value['stage']=='reviewed_source_union_inputs','Reviewed input-binding stage required')
 require(value['source_union_authorized']is True and value['admission_authorized']is False,'Explicit source-only review authorization required')
 exact(value['producers'],('pdf','html'),'producer bindings')
 refs=[]
 for kind,p in value['producers'].items():
  exact(p,('directory','plan_sha256','manifest_sha256','independent_reviews'),kind+' binding')
  require(type(p['directory'])is str and p['directory'].startswith('outputs/') and '..' not in Path(p['directory']).parts,'Local versioned producer directory required')
  for k in ('plan_sha256','manifest_sha256'):base.expect_sha(p[k])
  require(type(p['independent_reviews'])is list and p['independent_reviews'],'Independent '+kind+' reviews required')
  for review in p['independent_reviews']:
   exact(review,('path','sha256'),'independent review');refs.append(review)
 review=value['accounting_review'];exact(review,('path','sha256'),'accounting review');refs.append(review)
 paths=[]
 for ref in refs:
  require(type(ref['path'])is str and bool(ref['path']),'Review path required');base.expect_sha(ref['sha256']);paths.append(ref['path'])
 require(len(paths)==len(set(paths)),'Distinct independent review artifacts required')
 return refs


def bundle(root,kind,plan_sha,manifest_sha,directory=None):
 directory=directory or DIRECTORIES[kind]
 plan=base.read(root,directory+'/PLAN.json',plan_sha)
 require(plan['runtime']==sys.version,'Producer runtime mismatch: '+kind)
 require(plan['stage']==STAGES[kind] if kind=='baseline' else type(plan.get('stage'))is str and plan['stage'].startswith('frozen_'),'Producer frozen stage required: '+kind)
 if kind=='html':
  require(all(type(plan.get(k))is int and plan[k]==0 for k in ('financial_calculations','model_fits','ranks_computed','returns_computed','fact_admissions')),'HTML source-only plan required')
 else:
  require(all(plan[k]is False for k in ('financial_calculations_permitted','admission_permitted','network_requests_permitted')),'Producer source-only plan required')
 base.safe.verify_hashes(root,plan['source_hashes']);base.safe.verify_hashes(root,plan['source_code_hashes'])
 manifest=base.read(root,directory+'/MANIFEST.json',manifest_sha)
 require(manifest['plan_sha256']==plan_sha,'Producer manifest plan mismatch')
 exact(manifest['output_hashes'],('RAW_FACTS.json','AUDIT.json'),'producer manifest outputs')
 hashes={directory+'/PLAN.json':plan_sha,directory+'/MANIFEST.json':manifest_sha}
 for name,sha in manifest['output_hashes'].items():
  base.expect_sha(sha);hashes[directory+'/'+name]=sha
 data=base.read(root,directory+'/RAW_FACTS.json',hashes[directory+'/RAW_FACTS.json'])
 audit=base.read(root,directory+'/AUDIT.json',hashes[directory+'/AUDIT.json'])
 require(data['plan_sha256']==audit['plan_sha256']==plan_sha,'Producer output plan mismatch')
 require(audit.get('passed')is True and audit['data_sha256']==hashes[directory+'/RAW_FACTS.json'],'Producer audit mismatch')
 source_only(data)
 require(audit['counts']==data['counts'],'Producer audit coverage mismatch')
 return dict(plan=plan,data=data,audit=audit,manifest=manifest,hashes=hashes)


def review_contract(root,bindings,bundles,reviews,accounting):
 for kind in ('pdf','html'):
  pins={}
  for ref in bindings['producers'][kind]['independent_reviews']:
   record=reviews[ref['path']]
   require(record.get('passed')is True,'Passing independent review required: '+kind)
   for group in ('input_hashes','source_hashes','review_code_hashes'):
    if group in record:
     base.safe.verify_hashes(root,record[group])
     for path,sha in record[group].items():
      require(path not in pins or pins[path]==sha,'Independent review pin conflict');pins[path]=sha
   for field,name in (('data_sha256','RAW_FACTS.json'),('source_manifest_sha256','MANIFEST.json'),('plan_sha256','PLAN.json')):
    if field in record:
     path=bindings['producers'][kind]['directory']+'/'+name
     require(path not in pins or pins[path]==record[field],'Independent review direct pin conflict')
     pins[path]=record[field]
  for name in ('RAW_FACTS.json','MANIFEST.json'):
   path=bindings['producers'][kind]['directory']+'/'+name
   require(pins.get(path)==bundles[kind]['hashes'][path],'Independent review is not bound to current producer '+name)
 require(accounting.get('stage')=='source_only_field_scoped_accounting_review' and accounting.get('source_only')is True,'Separate source-only accounting review required')
 require(accounting.get('source_admissions')==0 and accounting.get('plan_sha256')==BASELINE_PINS[DIRECTORIES['baseline']+'/PLAN.json'],'Accounting review baseline/admission mismatch')
 base.safe.verify_hashes(root,accounting['source_hashes'])


def resolve_inputs(root,binding_path,binding_sha):
 base.expect_sha(binding_sha);bindings=base.read(root,binding_path,binding_sha);refs=binding_contract(bindings)
 base.safe.verify_hashes(root,BASELINE_PINS)
 directory=DIRECTORIES['baseline']
 bundles={'baseline':bundle(root,'baseline',BASELINE_PINS[directory+'/PLAN.json'],BASELINE_PINS[directory+'/MANIFEST.json'])}
 for kind in ('pdf','html'):
  b=bindings['producers'][kind];bundles[kind]=bundle(root,kind,b['plan_sha256'],b['manifest_sha256'],b['directory'])
 reviews={ref['path']:base.read(root,ref['path'],ref['sha256']) for ref in refs}
 accounting=reviews[bindings['accounting_review']['path']]
 review_contract(root,bindings,bundles,reviews,accounting)
 hashes=dict(BASELINE_PINS);hashes[binding_path]=binding_sha
 for b in bundles.values():hashes.update(b['hashes'])
 for ref in refs:hashes[ref['path']]=ref['sha256']
 return bindings,bundles,reviews,accounting,hashes


def metadata_contract(baseline_plan,baseline,pdf,html,accounting,expected_events=44,expected_existing=157):
 """No monetary comparison or union: verify exact IDs, provenance and null masks."""
 specs=indexed(baseline_plan['events'],'baseline plan')
 originals=indexed(baseline['events'],'baseline raw')
 producer_maps={'pdf':indexed(pdf['events'],'PDF producer'),'html':indexed(html['events'],'HTML producer')}
 require(len(specs)==expected_events and set(specs)==set(originals),'Complete fixed baseline census required')
 require(counted(baseline)==expected_existing,'Frozen baseline existing-cell count changed')
 require(not set(producer_maps['pdf']).intersection(producer_maps['html']),'Producers overlap')
 expected_pdf={key for key,e in specs.items() if e['symbol']=='AAPL'}
 require(set(producer_maps['pdf'])==expected_pdf and set(producer_maps['html'])==set(specs)-expected_pdf,'Exact issuer partition required')
 if expected_events==44:require(len(expected_pdf)==10 and len(producer_maps['html'])==34,'Exact ten/34 partition required')
 accounting_events=indexed(accounting['events'],'accounting review')
 require(set(accounting_events)==set(specs),'Accounting review must retain fixed full census')
 for data in (baseline,pdf,html):source_only(data)
 for key,spec in specs.items():
  kind='pdf' if key in expected_pdf else 'html';producer=producer_maps[kind][key];old=originals[key]
  metadata={k:v for k,v in spec.items() if k!='locator_inventory'}
  for name,record in (('baseline',old),('producer',producer)):
   require(all(k in record and record[k]==v for k,v in metadata.items()),name+' changed frozen event metadata: '+key)
  require(not ADDED_EVENT_KEYS.intersection(producer),'Producer collides with union metadata keys')
  for name,record in (('baseline',old),('producer',producer)):
   gaps={(g.get('side'),g.get('field')) for g in record['gaps']}
   for side in SIDES:
    for field in FIELDS:
     cell=record['fields'][side][field]
     if cell is None:
      require((side,field)in gaps,name+' missing position without explicit gap')
     else:
      require(cell['source']==metadata['source'],name+' cell source differs from event')
      require(cell['period']==metadata['periods'][side],name+' cell period differs from event')
      require(cell['available_from']==metadata['available_from'],name+' cell availability differs from event')
  review=accounting_events[key]
  for name in ADMISSIONS:require(review.get(name)is False,'Accounting review is not admission')
  require(review['inherited_rejection_codes']==metadata['inherited_rejection_codes'],'Accounting review lost inherited exclusions')
  require(review['periods']==metadata['periods'],'Accounting review changed periods')
 return specs,originals,producer_maps,accounting_events


def same_legacy_cell(old,new):
 require(new is not None,'Producer removed an existing v1 value; fallback forbidden')
 require(base.safe.decimal(old['value_decimal'])==base.safe.decimal(new['value_decimal']),'Existing monetary value conflict')
 for key in ('scale','currency_literal','currency_iso','period','available_from','source'):
  require(key in new and old[key]==new[key],'Existing cell provenance conflict: '+key)


def completeness(event):
 missing=[s+'.'+f for s,f in REQUIRED if event['fields'][s][f]is None]
 return dict(required_positions=[s+'.'+f for s,f in REQUIRED],missing_required_positions=missing,
  all_four_required_positions_present=not missing,
  meaning='Raw presence only; no accounting comparability, eligibility or admission conclusion.',
  comparison_flow_cells_retained_but_not_required=['comparison.operating_cash_flow','comparison.net_income'])


def consolidate(baseline_plan,baseline,pdf,html,accounting,bindings,expected_events=44,expected_existing=157):
 specs,oldmap,producers,accounting_events=metadata_contract(baseline_plan,baseline,pdf,html,accounting,expected_events,expected_existing)
 rows=[];coverage=[];preserved=0
 for spec in baseline_plan['events']:
  key=spec['event_id'];kind='pdf' if spec['symbol']=='AAPL' else 'html';producer=producers[kind][key];old=oldmap[key]
  result=deepcopy(producer);legacy_cells={s:{} for s in SIDES}
  for side in SIDES:
   for field in FIELDS:
    before=old['fields'][side][field];after=producer['fields'][side][field]
    if before is not None:
     same_legacy_cell(before,after);preserved+=1;legacy_cells[side][field]=deepcopy(before)
    if after is not None:
     # Validate decimal representation; no factor or monetary arithmetic.
     base.safe.decimal(after['value_decimal'])
    coverage.append(dict(event_id=key,side=side,field=field,present=after is not None,producer=kind,
     present_in_frozen_v1=before is not None,required_for_current_concept=(side,field)in REQUIRED,
     fact_admitted=False,model_admitted=False,strategy_admitted=False))
  result['required_field_completeness']=completeness(producer)
  result['union_provenance']=dict(producer=kind,
   producer_manifest_sha256=bindings['producers'][kind]['manifest_sha256'],
   producer_event_sha256=base.safe.digest(base.safe.encoded(producer)),
   baseline_metadata={k:deepcopy(v) for k,v in spec.items() if k!='locator_inventory'},
   frozen_v1_existing_cells=legacy_cells,
   frozen_v1_gaps=deepcopy(old['gaps']),frozen_v1_metadata_gaps=deepcopy(old['metadata_gaps']),
   frozen_v1_accounting_review_flags=deepcopy(old['accounting_review_flags']),
   independent_reviews=deepcopy(bindings['producers'][kind]['independent_reviews']),
   accounting_review_reference=deepcopy(bindings['accounting_review']),
   accounting_event_review=deepcopy(accounting_events[key]))
  require({k:v for k,v in result.items() if k not in ADDED_EVENT_KEYS}==producer,'Producer event changed during consolidation')
  rows.append(result)
 require(preserved==expected_existing,'All existing baseline cells must be retained')
 require(len(coverage)==expected_events*6,'Full raw position census required')
 return rows,coverage,preserved


def schema():
 return dict(schema_version=1,stage='source_only_union_not_admission',events=44,positions=264,pdf_events=10,html_events=34,
  preserved_v1_nonnull_cells=157,required_positions=[s+'.'+f for s,f in REQUIRED],
  selection='Fixed disjoint producer partition; never first-match, fallback or cross-source selection. Null stays null, except an existing v1 non-null becoming null fails the entire build.',
  conservation='Existing decimal monetary values exactly equal, with unchanged scale, literal/ISO currency, source, period and availability. Original v1 cell records retained in distinct provenance.',
  preservation='Producer event deep equality after removing exactly union_provenance and required_field_completeness. All producer flags/gaps remain untouched; baseline/accounting/independent reviews are separate provenance.',
  accounting='Raw field presence is not eligibility. Preserve all 44 accounting events, complete findings, six inherited event exclusions and unresolved comparisons.',
  prohibitions=['No source extraction','No sums or ratios','No signs/ranks/factors','No model or return calculations','No data or strategy admission'],
  binding_document=dict(schema_version=1,stage='reviewed_source_union_inputs',source_union_authorized=True,admission_authorized=False,
   producers={kind:dict(directory='<versioned-output-directory>',plan_sha256='<64hex>',manifest_sha256='<64hex>',independent_reviews=[dict(path='<local-relative-review.json>',sha256='<64hex>')]) for kind in ('pdf','html')},
   accounting_review=dict(path='work/cashflow_accounting_review_v1.json',sha256='<64hex>')))


def draft(root,binding_path,binding_sha):
 require(not base.safe.local(root,OUT+'/PLAN.json').exists(),'Frozen union is immutable')
 tests=base.read(root,OUT+'/TESTS.json');code=code_hashes(root)
 require(tests.get('passed')is True and tests.get('synthetic_only')is True and tests.get('source_code_hashes')==code,'Current passing synthetic tests required')
 require(all(type(tests.get(k))is int and tests[k]==0 for k in ('failures','errors','real_source_unions')),'Clean synthetic union scope required')
 bindings,bundles,reviews,accounting,hashes=resolve_inputs(root,binding_path,binding_sha)
 metadata_contract(bundles['baseline']['plan'],bundles['baseline']['data'],bundles['pdf']['data'],bundles['html']['data'],accounting)
 hashes[OUT+'/TESTS.json']=base.sha(root,OUT+'/TESTS.json')
 plan=dict(schema_version=1,stage='draft_cashflow_source_union_requires_review',created_utc=datetime.now(timezone.utc).isoformat(),runtime=sys.version,
  source_hashes=hashes,source_code_hashes=code,binding_path=binding_path,binding_sha256=binding_sha,policy=schema(),
  event_ids=[e['event_id'] for e in bundles['baseline']['plan']['events']],expected_events=44,expected_positions=264,expected_existing_cells=157,
  financial_calculations_permitted=False,admission_permitted=False,network_requests_permitted=False)
 base.publish(root,OUT+'/PLAN_DRAFT.json',plan)
 return dict(draft_sha256=base.sha(root,OUT+'/PLAN_DRAFT.json'),events=44,positions=264,actual_source_unions=0)


def freeze(root,draft_sha):
 base.expect_sha(draft_sha);plan=base.read(root,OUT+'/PLAN_DRAFT.json',draft_sha)
 require(plan['stage']=='draft_cashflow_source_union_requires_review' and plan['runtime']==sys.version,'Reviewed current draft required')
 require(plan['source_code_hashes']==code_hashes(root),'Draft code changed')
 base.safe.verify_hashes(root,plan['source_hashes'])
 plan['stage']='frozen_cashflow_source_union_plan';plan['reviewed_draft_sha256']=draft_sha
 plan['source_hashes'][OUT+'/PLAN_DRAFT.json']=draft_sha
 base.publish(root,OUT+'/PLAN.json',plan)
 return dict(plan_sha256=base.sha(root,OUT+'/PLAN.json'),actual_source_unions=0)


def products(root,plan_sha):
 base.expect_sha(plan_sha);plan=base.read(root,OUT+'/PLAN.json',plan_sha)
 require(plan['stage']=='frozen_cashflow_source_union_plan' and plan['runtime']==sys.version,'Frozen plan/runtime required')
 require(plan['source_code_hashes']==code_hashes(root),'Frozen code changed')
 require(all(plan[k]is False for k in ('financial_calculations_permitted','admission_permitted','network_requests_permitted')),'Source-only union required')
 require((plan['expected_events'],plan['expected_positions'],plan['expected_existing_cells'])==(44,264,157),'Fixed full census required')
 base.safe.verify_hashes(root,plan['source_hashes']);base.safe.verify_hashes(root,plan['source_code_hashes'])
 bindings,bundles,reviews,accounting,hashes=resolve_inputs(root,plan['binding_path'],plan['binding_sha256'])
 require(all(plan['source_hashes'].get(p)==h for p,h in hashes.items()),'Unregistered input closure')
 require(plan['event_ids']==[e['event_id'] for e in bundles['baseline']['plan']['events']],'Frozen event order changed')
 rows,coverage,preserved=consolidate(bundles['baseline']['plan'],bundles['baseline']['data'],bundles['pdf']['data'],bundles['html']['data'],accounting,bindings)
 count=sum(p['present'] for p in coverage);complete=sum(e['required_field_completeness']['all_four_required_positions_present'] for e in rows)
 counts=dict(events=44,requested_cells=264,extracted_cells=count,gap_cells=264-count,existing_v1_cells_preserved=preserved,events_with_four_required_positions=complete)
 data=dict(schema_version=1,stage='raw_source_union_not_admitted',plan_sha256=plan_sha,events=rows,counts=counts,
  review_provenance=dict(bindings=bindings,review_records=reviews,accounting_review=accounting),
  first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False,
  **{key:0 for key in ZERO_COUNTS})
 positions=dict(schema_version=1,stage='raw_coverage_not_eligibility',plan_sha256=plan_sha,counts=counts,positions=coverage,
  event_completeness=[dict(event_id=e['event_id'],**e['required_field_completeness']) for e in rows])
 audit=dict(schema_version=1,stage='source_union_process_audit_not_admission',passed=True,plan_sha256=plan_sha,counts=counts,
  meaning='Exact fixed census, producer records and baseline monetary provenance conserved; raw completeness does not admit any field or event.',
  data_sha256=base.safe.digest(base.safe.encoded(data)),coverage_sha256=base.safe.digest(base.safe.encoded(positions)),
  all_producer_events_deep_equal=True,all_existing_v1_cells_preserved=True,all_accounting_flags_preserved=True,
  source_hashes=plan['source_hashes'],source_code_hashes=plan['source_code_hashes'],network_requests=0,
  **{key:0 for key in ZERO_COUNTS})
 base.safe.verify_hashes(root,plan['source_hashes'])
 return {'RAW_FACTS.json':data,'COVERAGE.json':positions,'AUDIT.json':audit}


def build(root,plan_sha,verify=False):
 values=products(root,plan_sha)
 manifest=dict(plan_sha256=plan_sha,output_hashes={name:base.safe.digest(base.safe.encoded(value)) for name,value in values.items()})
 values['MANIFEST.json']=manifest
 for name,value in values.items():
  if verify:require(base.read(root,OUT+'/'+name)==value,'Deterministic union reproduction mismatch')
  else:base.publish(root,OUT+'/'+name,value)
 return dict(passed=True,verified=verify,counts=values['RAW_FACTS.json']['counts'],output_hashes=manifest['output_hashes'])


def main():
 p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
 for flag in ('schema','draft-plan','freeze-plan','build','verify'):mode.add_argument('--'+flag,action='store_true')
 p.add_argument('--bindings');p.add_argument('--bindings-sha');p.add_argument('--draft-sha');p.add_argument('--plan-sha');a=p.parse_args()
 result=schema() if a.schema else draft(ROOT,a.bindings,a.bindings_sha) if a.draft_plan else freeze(ROOT,a.draft_sha) if a.freeze_plan else build(ROOT,a.plan_sha,a.verify)
 print(base.safe.json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
