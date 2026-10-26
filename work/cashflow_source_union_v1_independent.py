"""Independent standard-library union verification; no producer imports."""
import argparse
from copy import deepcopy
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
OUT='outputs/monthly_cashflow_source_union_v1'
BASE='outputs/monthly_cashflow_source_preflight_v1'
SELF='work/cashflow_source_union_v1_independent.py'
BASE_PINS={
 BASE+'/PLAN.json':'4161a752cfac3039ea733b000cab647b8c77c48562ef6f497d109aefc7be1472',
 BASE+'/RAW_FACTS.json':'4f6d6e694195a4986f911f7ba1defc24efc515b49aa510b5c4be61df78c45c87',
 BASE+'/AUDIT.json':'43e908b15cf4ceb5fdfce700d74c6ff3788b906cff8f485c8c5e55026161c143',
 BASE+'/MANIFEST.json':'ee636a289dab86fe4758b60f0073bbad5be2ecd327f2cce552235719ec10a068'}
SIDES=('current','comparison')
FIELDS=('operating_cash_flow','net_income','total_assets')
REQUIRED=(('current','operating_cash_flow'),('current','net_income'),('current','total_assets'),('comparison','total_assets'))
EXTRA={'union_provenance','required_field_completeness'}
FLAGS=('fact_admitted','model_admitted','strategy_admitted')
CERTS=('first_public_version_verified','complete_correction_history_verified','certified_daily_security_master')
ZERO=('fact_admissions','model_admissions','strategy_admissions','financial_calculations','ratios_computed','signs_computed','ranks_computed','returns_computed','model_fits')

def need(ok,why):
 if not ok:raise ValueError(why)

def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
def digest(raw):return hashlib.sha256(raw).hexdigest()
def sha_string(value):need(type(value)is str and re.fullmatch('[0-9a-f]{64}',value),'Explicit lowercase SHA256 required')
def amount(value):
 need(type(value)is str and re.fullmatch(r'-?(?:0|[1-9]\d*)(?:\.\d+)?',value),'Plain finite decimal string required')
 result=Decimal(value);need(result.is_finite(),'Finite decimal required');return result

def unique(rows):
 need(type(rows)is list,'Event list required');out={}
 for row in rows:
  key=row.get('event_id');need(type(key)is str and key and key not in out,'Unique event identifiers required');out[key]=row
 return out

class Evidence:
 def __init__(self,root):self.root=Path(root).resolve();self.hashes={};self.states={}
 def path(self,name):
  need(type(name)is str and name and '..'not in Path(name).parts,'Bound local path required')
  need(not any(x in name.lower()for x in ('onedrive','cloudstorage')),'Cloud paths forbidden')
  p=Path(name);p=p if p.is_absolute()else self.root/p
  allowed=p.is_relative_to(self.root)or str(p).startswith('/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/')
  need(allowed and p.resolve()==p,'Unbound external path or symlink')
  return p
 def hash(self,name,expected=None):
  p=self.path(name);s=p.stat();need(stat.S_ISREG(s.st_mode),'Regular file required')
  state=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
  if name not in self.hashes:
   h=hashlib.sha256()
   with p.open('rb')as f:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
   result=h.hexdigest();after=p.stat()
   need(state==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'Input changed during hash')
   self.hashes[name]=result;self.states[name]=state
  else:
   need(self.states[name]==state,'Input changed during independent review');result=self.hashes[name]
  if expected is not None:sha_string(expected);need(result==expected,'SHA mismatch: '+name)
  return result
 def read(self,name,expected=None):
  self.hash(name,expected)
  def pairs(items):
   value={}
   for k,v in items:need(k not in value,'Duplicate JSON key');value[k]=v
   return value
  def bad(value):raise ValueError('Nonfinite JSON token '+value)
  return json.loads(self.path(name).read_bytes(),object_pairs_hook=pairs,parse_constant=bad)
 def pins(self,group):
  need(type(group)is dict and group,'Nonempty hash closure required')
  for p,h in group.items():self.hash(p,h)
 def stable(self):
  for name in list(self.hashes):self.hash(name,self.hashes[name])

def check_flags(data):
 for key in ZERO:need(type(data.get(key))is int and data[key]==0,'Zero source-only counter: '+key)
 for key in CERTS:
  if key in data:need(data[key]is False,'Unexpected certification')
 for event in data['events']:
  for key in FLAGS:need(event.get(key)is False,'Event unexpectedly admitted')
  for key in CERTS:
   if key in event:need(event[key]is False,'Event unexpectedly certified')
  need(set(event['fields'])==set(SIDES),'Exact two field sides required')
  for side in SIDES:
   need(set(event['fields'][side])==set(FIELDS),'Exact three fields required')
   for cell in event['fields'][side].values():
    if cell is None:continue
    need(cell.get('fact_admitted')is False,'Cell unexpectedly admitted')
    for key in FLAGS:
     if key in cell:need(cell[key]is False,'Cell unexpectedly admitted')
    amount(cell['value_decimal'])

def expected_completeness(event):
 missing=[s+'.'+f for s,f in REQUIRED if event['fields'][s][f]is None]
 return dict(required_positions=[s+'.'+f for s,f in REQUIRED],missing_required_positions=missing,
  all_four_required_positions_present=not missing,
  meaning='Raw presence only; no accounting comparability, eligibility or admission conclusion.',
  comparison_flow_cells_retained_but_not_required=['comparison.operating_cash_flow','comparison.net_income'])

def validate_records(baseline_plan,baseline,pdf,html,merged,coverage,bindings,accounting,expected_events=44,expected_old=157):
 specs=unique(baseline_plan['events']);old=unique(baseline['events']);left=unique(pdf['events']);right=unique(html['events']);actual=unique(merged['events']);acc=unique(accounting['events'])
 need(len(specs)==expected_events and set(specs)==set(old)==set(actual)==set(acc),'Complete baseline/accounting/union census')
 need(not(set(left)&set(right))and set(left)|set(right)==set(specs),'Disjoint complete producer census')
 need(set(left)=={k for k,v in specs.items()if v['symbol']=='AAPL'},'Exact PDF issuer partition')
 need([e['event_id']for e in merged['events']]==list(specs),'Canonical event order')
 check_flags(merged)
 expected_positions=[];old_count=0;complete=0;gap_positions=[]
 for key,spec in specs.items():
  kind='pdf'if key in left else'html';producer=(left if kind=='pdf'else right)[key];event=actual[key];prior=old[key]
  need(not(EXTRA&set(producer)),'Producer collides with documented union metadata')
  need(set(event)==set(producer)|EXTRA,'Unexpected union-only event field')
  need({k:v for k,v in event.items()if k not in EXTRA}==producer,'Producer event was modified: '+key)
  metadata={k:v for k,v in spec.items()if k!='locator_inventory'}
  need(all(event.get(k)==v for k,v in metadata.items()),'Frozen event metadata changed')
  provenance=event['union_provenance']
  need(provenance['producer']==kind,'Producer route mismatch')
  need(provenance['producer_manifest_sha256']==bindings['producers'][kind]['manifest_sha256'],'Producer manifest provenance')
  need(provenance['producer_event_sha256']==digest(encoded(producer)),'Producer event digest mismatch')
  need(provenance['baseline_metadata']==metadata,'Baseline metadata provenance')
  for name,old_name in (('frozen_v1_gaps','gaps'),('frozen_v1_metadata_gaps','metadata_gaps'),('frozen_v1_accounting_review_flags','accounting_review_flags')):
   need(provenance[name]==prior[old_name],'Original V1 gap/accounting evidence lost')
  need(provenance['independent_reviews']==bindings['producers'][kind]['independent_reviews'],'Independent review provenance mismatch')
  need(provenance['accounting_review_reference']==bindings['accounting_review'],'Accounting reference mismatch')
  need(provenance['accounting_event_review']==acc[key],'Accounting event changed')
  need(event['inherited_rejection_codes']==prior['inherited_rejection_codes']==acc[key]['inherited_rejection_codes'],'Inherited rejection lost')
  for flag in FLAGS:need(acc[key].get(flag)is False,'Accounting review admits event')
  need(acc[key]['periods']==event['periods'],'Accounting period mismatch')
  comp=expected_completeness(event);need(event['required_field_completeness']==comp,'Required-field completeness mismatch')
  complete+=comp['all_four_required_positions_present'];legacy={s:{}for s in SIDES}
  nulls={(g['side'],g['field'])for g in event['gaps']}
  for side in SIDES:
   for field in FIELDS:
    before=prior['fields'][side][field];after=event['fields'][side][field]
    if after is None:
     need((side,field)in nulls,'Null lacks explicit gap');gap_positions.append((key,side,field))
    else:
     need(after['source']==event['source']and after['period']==event['periods'][side]and after['available_from']==event['available_from'],'Cell source/period/availability mismatch')
    if before is not None:
     old_count+=1;need(after is not None and amount(before['value_decimal'])==amount(after['value_decimal']),'Existing V1 value removed or changed')
     for name in ('scale','currency_literal','currency_iso','period','available_from','source'):
      need(after[name]==before[name],'Existing V1 provenance changed: '+name)
     legacy[side][field]=before
    expected_positions.append(dict(event_id=key,side=side,field=field,present=after is not None,producer=kind,present_in_frozen_v1=before is not None,
     required_for_current_concept=(side,field)in REQUIRED,fact_admitted=False,model_admitted=False,strategy_admitted=False))
  need(provenance['frozen_v1_existing_cells']==legacy,'Original full V1 cell dictionaries not exactly preserved')
 need(old_count==expected_old,'Unexpected baseline nonnull count')
 need(coverage['positions']==expected_positions,'Full position matrix mismatch')
 need(coverage['event_completeness']==[dict(event_id=e['event_id'],**expected_completeness(e))for e in merged['events']],'Coverage completeness matrix mismatch')
 n=len(expected_positions);present=sum(x['present']for x in expected_positions)
 counts=dict(events=expected_events,requested_cells=n,extracted_cells=present,gap_cells=n-present,existing_v1_cells_preserved=old_count,events_with_four_required_positions=complete)
 need(merged['counts']==coverage['counts']==counts,'Union recomputed counts mismatch')
 return counts

def read_bundle(evidence,directory,plan_sha,manifest_sha):
 plan=evidence.read(directory+'/PLAN.json',plan_sha);manifest=evidence.read(directory+'/MANIFEST.json',manifest_sha)
 need(plan['runtime']==sys.version and plan['stage'].startswith('frozen_'),'Frozen current runtime required')
 for key in ('financial_calculations_permitted','admission_permitted','network_requests_permitted'):
  if key in plan:need(plan[key]is False,'Producer plan permits non-source operations')
 evidence.pins(plan['source_hashes']);evidence.pins(plan['source_code_hashes'])
 if plan.get('pdf_runtime_hashes'):evidence.pins(plan['pdf_runtime_hashes'])
 need(manifest['plan_sha256']==plan_sha,'Producer plan/manifest mismatch')
 need(set(manifest['output_hashes'])=={'RAW_FACTS.json','AUDIT.json'},'Exact producer output manifest')
 data=evidence.read(directory+'/RAW_FACTS.json',manifest['output_hashes']['RAW_FACTS.json']);audit=evidence.read(directory+'/AUDIT.json',manifest['output_hashes']['AUDIT.json'])
 need(data['plan_sha256']==audit['plan_sha256']==plan_sha,'Producer output plan binding')
 need(audit['passed']is True and audit['data_sha256']==manifest['output_hashes']['RAW_FACTS.json'],'Producer audit binding')
 need(audit['counts']==data['counts'],'Producer audit counts differ')
 check_flags(data)
 count=sum(c is not None for row in data['events']for side in SIDES for c in row['fields'][side].values())
 need(all(data['counts'][k]==v for k,v in dict(events=len(data['events']),requested_cells=len(data['events'])*6,extracted_cells=count,gap_cells=len(data['events'])*6-count).items()),'Producer counts not recomputed')
 return plan,data,audit,manifest

def execute(root,plan_sha):
 sha_string(plan_sha);e=Evidence(root);plan=e.read(OUT+'/PLAN.json',plan_sha)
 need(plan['stage']=='frozen_cashflow_source_union_plan'and plan['runtime']==sys.version,'Frozen union plan/runtime')
 need(all(plan[k]is False for k in ('financial_calculations_permitted','admission_permitted','network_requests_permitted')),'Source-only union plan')
 need((plan['expected_events'],plan['expected_positions'],plan['expected_existing_cells'])==(44,264,157),'Fixed 44/264/157 plan')
 e.pins(plan['source_hashes']);e.pins(plan['source_code_hashes']);e.pins(BASE_PINS)
 manifest=e.read(OUT+'/MANIFEST.json')
 need(manifest['plan_sha256']==plan_sha and set(manifest['output_hashes'])=={'RAW_FACTS.json','COVERAGE.json','AUDIT.json'},'Exact union manifest')
 records={name:e.read(OUT+'/'+name,h)for name,h in manifest['output_hashes'].items()}
 data=records['RAW_FACTS.json'];coverage=records['COVERAGE.json'];audit=records['AUDIT.json']
 need(all(x['plan_sha256']==plan_sha for x in records.values()),'Union output plan binding')
 need(audit['passed']is True and audit['data_sha256']==manifest['output_hashes']['RAW_FACTS.json']and audit['coverage_sha256']==manifest['output_hashes']['COVERAGE.json'],'Union audit output hashes')
 check_flags(data)
 for key in ZERO:need(type(audit.get(key))is int and audit[key]==0,'Union audit source-only counter')
 need(audit['source_hashes']==plan['source_hashes']and audit['source_code_hashes']==plan['source_code_hashes'],'Union audit source closure')
 bindings=e.read(plan['binding_path'],plan['binding_sha256'])
 need(bindings['stage']=='reviewed_source_union_inputs'and bindings['source_union_authorized']is True and bindings['admission_authorized']is False,'Source-only reviewed input authorization')
 need(set(bindings['producers'])=={'pdf','html'},'Two exact input producers')
 baseline=read_bundle(e,BASE,BASE_PINS[BASE+'/PLAN.json'],BASE_PINS[BASE+'/MANIFEST.json'])
 bundles={};reviews={}
 for kind,binding in bindings['producers'].items():
  directory=binding['directory'];bundles[kind]=read_bundle(e,directory,binding['plan_sha256'],binding['manifest_sha256'])
  direct={directory+'/PLAN.json':binding['plan_sha256'],directory+'/MANIFEST.json':binding['manifest_sha256']}
  direct.update({directory+'/'+name:h for name,h in bundles[kind][3]['output_hashes'].items()})
  need(all(plan['source_hashes'].get(p)==h for p,h in direct.items()),'Producer artifacts omitted from union frozen closure')
  review_pins={}
  for ref in binding['independent_reviews']:
   need(plan['source_hashes'].get(ref['path'])==ref['sha256'],'Independent review omitted from union frozen closure')
   record=e.read(ref['path'],ref['sha256']);need(record.get('passed')is True,'Passing independent producer review required');reviews[ref['path']]=record
   for group in ('input_hashes','source_hashes','review_code_hashes'):
    if record.get(group):
     e.pins(record[group])
     for path,h in record[group].items():
      need(path not in review_pins or review_pins[path]==h,'Review pin conflict');review_pins[path]=h
   for field,name in (('data_sha256','RAW_FACTS.json'),('source_manifest_sha256','MANIFEST.json'),('plan_sha256','PLAN.json')):
    if field in record:
     path=directory+'/'+name;need(path not in review_pins or review_pins[path]==record[field],'Direct review pin conflict');review_pins[path]=record[field]
  for name,expected in (('RAW_FACTS.json',bundles[kind][3]['output_hashes']['RAW_FACTS.json']),('MANIFEST.json',binding['manifest_sha256'])):
   need(review_pins.get(directory+'/'+name)==expected,'Independent review must bind current producer data/manifest')
 ref=bindings['accounting_review'];accounting=e.read(ref['path'],ref['sha256']);reviews[ref['path']]=accounting
 need(plan['source_hashes'].get(ref['path'])==ref['sha256']and plan['source_hashes'].get(plan['binding_path'])==plan['binding_sha256'],'Review/binding omitted from union frozen closure')
 need(accounting['stage']=='source_only_field_scoped_accounting_review'and accounting['source_only']is True and type(accounting['source_admissions'])is int and accounting['source_admissions']==0,'Source-only accounting review')
 need(accounting['plan_sha256']==BASE_PINS[BASE+'/PLAN.json'],'Accounting baseline mismatch');e.pins(accounting['source_hashes'])
 need(data['review_provenance']==dict(bindings=bindings,review_records=reviews,accounting_review=accounting),'Review records changed in union')
 counts=validate_records(baseline[0],baseline[1],bundles['pdf'][1],bundles['html'][1],data,coverage,bindings,accounting)
 need(audit['counts']==counts and plan['event_ids']==[x['event_id']for x in baseline[0]['events']],'Plan event/audit count mismatch')
 e.hash(SELF);e.stable()
 receipt=dict(schema_version=1,stage='independent_cashflow_source_union_review',passed=True,plan_sha256=plan_sha,data_sha256=manifest['output_hashes']['RAW_FACTS.json'],
  source_manifest_sha256=e.hash(OUT+'/MANIFEST.json'),counts=counts,input_hashes=e.hashes,
  checks=dict(all_input_hashes_verified=True,exact_disjoint_44_events=True,producer_event_deep_equality=True,exact_full_v1_provenance_preserved=True,
   all264_positions_independently_rebuilt=True,required_completeness_recomputed=True,old_rejections_and_accounting_evidence_preserved=True,
   independent_current_producer_review_bindings_verified=True,all_source_only_counters_zero=True),
  financial_calculations=0,ratios_computed=0,signs_computed=0,model_fits=0,ranks_computed=0,returns_computed=0,fact_admissions=0,
  first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False,
  interpretation='Exact source-record union and raw completeness only. No field, accounting comparison, strategy or security-master admission.')
 output=e.path(OUT+'/INDEPENDENT_REVIEW.json');output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('xb')as f:f.write(encoded(receipt));f.flush();os.fsync(f.fileno())
 return dict(passed=True,counts=counts,review_sha256=digest(encoded(receipt)),verified_files=len(e.hashes))

class SelfTests(unittest.TestCase):
 def fixture(self):
  specs=[];original=[];acc=[];merged=[];positions=[]
  binding={'producers':{k:{'manifest_sha256':'a'*64,'independent_reviews':[]}for k in ('pdf','html')},'accounting_review':{'path':'accounting','sha256':'b'*64}}
  for i,symbol in enumerate(('AAPL','AMD')):
   source={'path':symbol+'.html','sha256':'c'*64};periods={s:{'period_end':'2024-12-31'}for s in SIDES}
   meta=dict(event_id=symbol,symbol=symbol,source=source,periods=periods,available_from='2025-01-10',inherited_rejection_codes=['unresolved'])
   specs.append(dict(meta,locator_inventory={}))
   row=dict(meta,fields={s:{f:dict(value_decimal='1.00',scale='1000',currency_literal='$',currency_iso=None,period=periods[s],source=source,available_from=meta['available_from'],fact_admitted=False)for f in FIELDS}for s in SIDES},
    gaps=[],metadata_gaps=[],accounting_review_flags=['unresolved'],**{k:False for k in FLAGS+CERTS})
   original.append(row);account=dict(event_id=symbol,periods=periods,inherited_rejection_codes=['unresolved'],**{k:False for k in FLAGS});acc.append(account)
   kind='pdf'if symbol=='AAPL'else'html';event=deepcopy(row);event['required_field_completeness']=expected_completeness(row)
   event['union_provenance']=dict(producer=kind,producer_manifest_sha256='a'*64,producer_event_sha256=digest(encoded(row)),baseline_metadata=meta,
    frozen_v1_existing_cells=deepcopy(row['fields']),frozen_v1_gaps=[],frozen_v1_metadata_gaps=[],frozen_v1_accounting_review_flags=['unresolved'],
    independent_reviews=[],accounting_review_reference=binding['accounting_review'],accounting_event_review=account)
   merged.append(event)
   positions.extend(dict(event_id=symbol,side=s,field=f,present=True,producer=kind,present_in_frozen_v1=True,required_for_current_concept=(s,f)in REQUIRED,**{k:False for k in FLAGS})for s in SIDES for f in FIELDS)
  counts=dict(events=2,requested_cells=12,extracted_cells=12,gap_cells=0,existing_v1_cells_preserved=12,events_with_four_required_positions=2)
  data=dict(events=merged,counts=counts,**{k:0 for k in ZERO})
  coverage=dict(positions=positions,counts=counts,event_completeness=[dict(event_id=e['event_id'],**expected_completeness(e))for e in merged])
  return [dict(events=specs),dict(events=original),dict(events=original[:1]),dict(events=original[1:]),data,coverage,binding,dict(events=acc)]
 def test_two_complete_producer_records(self):
  self.assertEqual(validate_records(*self.fixture(),expected_events=2,expected_old=12)['extracted_cells'],12)
 def test_extra_union_metadata_rejected(self):
  x=self.fixture();x[4]['events'][0]['extra']=1
  with self.assertRaises(ValueError):validate_records(*x,expected_events=2,expected_old=12)
 def test_modified_producer_record_rejected(self):
  x=self.fixture();x[4]['events'][0]['accounting_review_flags']=[]
  with self.assertRaises(ValueError):validate_records(*x,expected_events=2,expected_old=12)
 def test_old_cell_provenance_must_be_exact(self):
  x=self.fixture();x[4]['events'][0]['union_provenance']['frozen_v1_existing_cells']['current']['net_income']['value_decimal']='1'
  with self.assertRaises(ValueError):validate_records(*x,expected_events=2,expected_old=12)
 def test_coverage_count_tampering_rejected(self):
  x=self.fixture();x[5]['positions'][0]['required_for_current_concept']=False
  with self.assertRaises(ValueError):validate_records(*x,expected_events=2,expected_old=12)
 def test_required_completeness_tampering_rejected(self):
  x=self.fixture();x[4]['events'][0]['required_field_completeness']['all_four_required_positions_present']=False
  with self.assertRaises(ValueError):validate_records(*x,expected_events=2,expected_old=12)
 def test_exact_decimals(self):
  self.assertEqual(amount('-2.00'),Decimal('-2'));self.assertEqual(amount('0'),Decimal(0))
 def test_invalid_decimal_types(self):
  for x in (True,False,1,float('nan'),'NaN','Infinity','1e3','1,000'):
   with self.subTest(x=x),self.assertRaises(ValueError):amount(x)
 def test_duplicate_ids(self):
  with self.assertRaises(ValueError):unique([{'event_id':'x'},{'event_id':'x'}])
 def test_extra_metadata_key_set(self):self.assertEqual(EXTRA,{'union_provenance','required_field_completeness'})
 def test_required_excludes_prior_flow(self):
  event={'fields':{s:{f:1 for f in FIELDS}for s in SIDES}};event['fields']['comparison']['net_income']=None
  self.assertTrue(expected_completeness(event)['all_four_required_positions_present'])
 def test_missing_current_flow_incomplete(self):
  event={'fields':{s:{f:1 for f in FIELDS}for s in SIDES}};event['fields']['current']['net_income']=None
  self.assertFalse(expected_completeness(event)['all_four_required_positions_present'])
 def test_hash_mutation_detected(self):
  with tempfile.TemporaryDirectory()as tmp:
   p=Path(tmp,'source');p.write_bytes(b'first');e=Evidence(tmp);e.hash('source');p.write_bytes(b'second')
   with self.assertRaises(ValueError):e.stable()
 def test_wrong_hash_detected(self):
  with tempfile.TemporaryDirectory()as tmp:
   Path(tmp,'source').write_bytes(b'first')
   with self.assertRaises(ValueError):Evidence(tmp).hash('source','0'*64)
 def test_duplicate_json_rejected(self):
  with tempfile.TemporaryDirectory()as tmp:
   Path(tmp,'source').write_bytes(b'{"x":1,"x":2}')
   with self.assertRaises(ValueError):Evidence(tmp).read('source')
 def test_no_cloud_paths(self):
  with self.assertRaises(ValueError):Evidence(ROOT).path('/Users/shiweili/Library/CloudStorage/x')
 def test_bool_not_zero_counter(self):
  data={k:0 for k in ZERO};data.update(events=[]);data['returns_computed']=False
  with self.assertRaises(ValueError):check_flags(data)
 def test_sha_mandatory(self):
  for value in (None,'','A'*64,False):
   with self.assertRaises(ValueError):sha_string(value)

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true');parser.add_argument('--plan-sha')
 args=parser.parse_args()
 if args.self_test:
  result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SelfTests))
  raise SystemExit(0 if result.wasSuccessful()else 1)
 print(json.dumps(execute(ROOT,args.plan_sha),sort_keys=True))
