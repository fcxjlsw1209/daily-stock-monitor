"""Source-only exact union of two independently admitted same-report packages."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os

ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/monthly_same_report145_union'
METHOD = 'same_current_release_annual_comparison'
PACKAGES = {
 'aapl_amd': ('outputs/monthly_same_report144_admission', {'AAPL','AMD'}, 20),
 'msft_nvda': ('outputs/monthly_msft_nvda_same_report_admission_v1', {'MSFT','NVDA'}, 24),
}
OLD_DATA_SHA = 'c40788c6487af79c11cd68cb762d16ccdb220b0e5c59001a9033923aa28adce9'
OLD_AUDIT_SHA = 'bc310df597de8f5ec8ea433c11bb45cf09c8d4ce25e730282b141a41695a1e41'
CERTS = ('first_public_version_verified','complete_correction_history_verified','certified_daily_security_master')
COUNTERS = ('financial_calculations','model_fits','ranks_computed','returns_computed','network_requests')
CHECKS = ('complete_packages_preserved','unique_event_ids','disjoint_issuer_sets',
          'rejected_events_preserved','availability_unmodified','original_package_hashes_verified')
SOURCE_CHECKS = {
 'aapl_amd': ('all_eighty_values_and_scales_preserved','all_twenty_events_preserved','causal_identity_windows','four_rejection_sentinels_retained','latest_invalid_blocks_fallback','missing_AMD2019_not_fabricated','old_measurement_definition_not_merged','same_current_release_four_cells','strict_next_session_availability','two_annual_periods_checked'),
 'msft_nvda': ('all24_events_preserved','all96_cells_independently_verified','both_FY2015_rejections_retained','accounting_flags_resolved_only_by_evidence','same_current_release_four_cells','strict_next_session_availability','causal_identity_windows','latest_invalid_blocks_fallback'),
}

def require(ok, message):
 if not ok: raise ValueError(message)

def encoded(x):
 return (json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)+'\n').encode()

def sha(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
 return json.loads(Path(path).read_text())

def write_once(path, x):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(encoded(x));f.flush();os.fsync(f.fileno())

def path_at(root, value, primary=False):
 p=Path(value);p=(root/p if not p.is_absolute() else p).resolve()
 require('CloudStorage' not in str(p) and 'OneDrive' not in str(p),'Cloud input forbidden')
 if primary:require(p.is_relative_to(root),'Primary input outside local research root')
 return p

def validate_package(name, data, audit, data_sha):
 _,symbols,count=PACKAGES[name]
 require(data.get('method')==METHOD,'Different measurement method')
 require(audit.get('stage')=='conditional_source_admission' and audit.get('passed') is True,'Conditional source admission required')
 require(audit.get('data_sha256')==data_sha,'Dataset binding differs')
 require(set(audit.get('checks',{}))==set(SOURCE_CHECKS[name]) and all(audit['checks'][x] is True for x in SOURCE_CHECKS[name]),'Exact source checks required')
 for key in CERTS:require(data.get(key) is False and audit.get(key) is False,'Do not upgrade archival certification')
 for key in COUNTERS:require(type(audit.get(key)) is int and audit[key]==0,'Source-only counter')
 for key in ('model_admitted','strategy_admitted'):require(data.get(key) is False,'No strategy admission')
 events=data.get('events');require(isinstance(events,list) and len(events)==count,'Complete event census required')
 require({x.get('symbol') for x in events}==symbols,'Exact disjoint issuer package')
 ids=set()
 for event in events:
  eid=event.get('event_id');require(isinstance(eid,str) and eid and eid not in ids,'Unique event IDs required');ids.add(eid)
  require(event.get('method')==METHOD,'Event measurement differs')
  require(type(event.get('admitted')) is bool and isinstance(event.get('rejection_codes'),list),'Explicit admission and rejection state')
  require(event['admitted']==(not event['rejection_codes']),'Admission conflicts with rejection')
  require(event.get('security_id')==event['symbol'],'Security symbol mismatch')
  require(event.get('identity_valid_from')==event.get('available_from'),'Causal identity required')
  require(isinstance(event.get('available_from'),str) and event['available_from']>event.get('release_date',''),'Current release availability required')
  require(set(event.get('same_report',{}))=={'current','comparison'},'Two columns required')
  for period in event['same_report'].values():
   require('gross_profit' in period and 'assets' in period,'Four original values required')
 observed=dict(events=len(events),conditionally_admitted=sum(x['admitted'] for x in events),rejected=sum(not x['admitted'] for x in events),raw_values=4*len(events))
 for container in (data,audit):
  require(all(container.get('counts',{}).get(k)==v for k,v in observed.items()),'Package counts disagree with events')
 return events

def combine(items):
 """Pure merge with no monetary operations; reject overlap, never arbitrate."""
 require(set(items)==set(PACKAGES),'Two fixed packages required')
 events=[];seen=set();symbols=set();counts={}
 for name in PACKAGES:
  data,audit,expected=items[name];rows=validate_package(name,data,audit,expected)
  current={x['symbol'] for x in rows};require(not current & symbols,'Overlapping issuer packages');symbols|=current
  for event in rows:
   require(event['event_id'] not in seen,'Overlapping event');seen.add(event['event_id']);events.append(event)
  counts[name]=dict(events=len(rows),conditionally_admitted=sum(x['admitted'] for x in rows),rejected=sum(not x['admitted'] for x in rows),raw_values=4*len(rows))
 events=sorted(events,key=lambda x:(x['symbol'],x['available_from'],x['event_id']))
 return dict(schema_version=1,method=METHOD,policy_id='same_report145_union_v1',events=events,
  admission_scope='Exact source union only; independent conditional package assumptions preserved',
  counts=dict(events=len(events),conditionally_admitted=sum(x['admitted'] for x in events),rejected=sum(not x['admitted'] for x in events),raw_values=4*len(events)),
  package_counts=counts,assumptions={n:items[n][0].get('assumptions',{}) for n in PACKAGES},
  missing_publications={n:items[n][0].get('missing_publications',[]) for n in PACKAGES},
  model_admitted=False,strategy_admitted=False,ranks_computed=0,ratios_computed=0,returns_computed=0,signs_computed=0,
  **{x:False for x in CERTS})

def register(root, new_data_sha, new_audit_sha, tests_sha):
 out=root/OUT;require(not (out/'PLAN.json').exists(),'Union plan already exists')
 pins={};refs={}
 def add(value, expected=None):
  p=path_at(root,value);s=sha(p)
  if expected:require(s==expected,'Input SHA differs: '+str(p))
  key=str(p);require(key not in pins or pins[key]==s,'Conflicting hash');pins[key]=s
  return dict(path=str(p.relative_to(root)) if p.is_relative_to(root) else str(p),sha256=s)
 for name,(directory,_,_) in PACKAGES.items():
  expected=(OLD_DATA_SHA,OLD_AUDIT_SHA) if name=='aapl_amd' else (new_data_sha,new_audit_sha)
  dref=add(directory+'/DATA.json',expected[0]);aref=add(directory+'/AUDIT.json',expected[1])
  data=load(root/dref['path']);audit=load(root/aref['path']);validate_package(name,data,audit,dref['sha256'])
  for mapping in ('source_hashes','source_code_hashes'):
   require(isinstance(audit.get(mapping),dict) and audit[mapping],'Source closure required')
   for p,h in audit[mapping].items():add(p,h)
  for filename in ('PLAN.json','POLICY.json','TESTS.json','MANIFEST.json'):add(directory+'/'+filename)
  refs[name]=dict(data_path=dref['path'],data_sha256=dref['sha256'],audit_path=aref['path'],audit_sha256=aref['sha256'])
 for name in ('work/same_report145_union.py','work/test_same_report145_union.py','run_local.sh','local_run.py','local.sandbox.sb'):add(name)
 tref=add(OUT+'/TESTS.json',tests_sha);tests=load(root/tref['path'])
 require(tests.get('passed') is True and tests.get('tests',0)>0,'Passing union tests required')
 require(tests.get('source_sha256')==pins[str(root/'work/same_report145_union.py')] and tests.get('test_sha256')==pins[str(root/'work/test_same_report145_union.py')],'Test sources differ')
 if 'log' in tests:add(tests['log']['path'],tests['log']['sha256'])
 plan=dict(schema_version=1,registered_utc=datetime.now(timezone.utc).isoformat(),stage='source_union_predeclared',
  method=METHOD,packages=refs,source_hashes=pins,expected_events=44,expected_raw_values=176,
  rule='Preserve exact event dictionaries from both source packages, including all rejections; one union only, no monetary arithmetic.',
  financial_calculations=0,model_fits=0,ranks_computed=0,returns_computed=0,network_requests=0)
 write_once(out/'PLAN.json',plan);return sha(out/'PLAN.json')

def build(root, expected_plan, verify=False):
 out=root/OUT;require(sha(out/'PLAN.json')==expected_plan,'Union PLAN mismatch');plan=load(out/'PLAN.json')
 for p,h in plan['source_hashes'].items():require(sha(path_at(root,p))==h,'Source changed: '+p)
 items={}
 for name,ref in plan['packages'].items():
  dp=path_at(root,ref['data_path'],True);ap=path_at(root,ref['audit_path'],True)
  require(sha(dp)==ref['data_sha256'] and sha(ap)==ref['audit_sha256'],'Package reference mismatch')
  items[name]=(load(dp),load(ap),ref['data_sha256'])
 data=combine(items);dh=hashlib.sha256(encoded(data)).hexdigest()
 sources=dict(plan['source_hashes']);sources[str(out/'PLAN.json')]=expected_plan
 audit=dict(schema_version=1,stage='conditional_union_source_admission',passed=True,method=METHOD,
  data_sha256=dh,plan_sha256=expected_plan,packages=plan['packages'],counts=data['counts'],
  source_hashes=sources,source_code_hashes={str(root/x):sha(root/x) for x in ('work/same_report145_union.py','work/test_same_report145_union.py')},
  checks={x:True for x in CHECKS},limitations=data['assumptions'],**{x:0 for x in COUNTERS},**{x:False for x in CERTS})
 manifest=dict(plan_sha256=expected_plan,data_sha256=dh,audit_sha256=hashlib.sha256(encoded(audit)).hexdigest())
 for filename,value in (('DATA.json',data),('AUDIT.json',audit),('MANIFEST.json',manifest)):
  if verify:require((out/filename).read_bytes()==encoded(value),'Exact union replay mismatch: '+filename)
  else:write_once(out/filename,value)
 return dict(passed=True,verify=verify,**data['counts'],plan_sha256=expected_plan,data_sha256=dh,audit_sha256=manifest['audit_sha256'])

def main():
 p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
 mode.add_argument('--register',action='store_true');mode.add_argument('--build',action='store_true');mode.add_argument('--verify',action='store_true')
 p.add_argument('--new-data-sha');p.add_argument('--new-audit-sha');p.add_argument('--tests-sha');p.add_argument('--plan-sha');a=p.parse_args()
 if a.register:
  require(all((a.new_data_sha,a.new_audit_sha,a.tests_sha)),'Explicit admitted source/test hashes required')
  print(json.dumps(dict(plan_sha256=register(ROOT,a.new_data_sha,a.new_audit_sha,a.tests_sha))))
 else:print(json.dumps(build(ROOT,a.plan_sha,a.verify)))
if __name__=='__main__':main()
