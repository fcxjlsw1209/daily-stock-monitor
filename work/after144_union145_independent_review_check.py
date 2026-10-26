"""Independent source-only union verification; repeat runs verify the receipt."""
from pathlib import Path
import hashlib,json,datetime
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_same_report145_union'
REPORT=ROOT/'work/after144_union145_independent_review.json'
def path(x):
 p=Path(x);p=p if p.is_absolute() else ROOT/p
 assert p.resolve()==p and p.is_relative_to(ROOT),str(p)
 return p
def sha(x):return hashlib.sha256(path(x).read_bytes()).hexdigest()
def load(x):
 def pairs(items):
  o={}
  for k,v in items:
   assert k not in o,'duplicate JSON key';o[k]=v
  return o
 def bad(x):raise ValueError(x)
 return json.loads(path(x).read_bytes(),object_pairs_hook=pairs,parse_constant=bad)
def enc(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
def normalized(m):return {str(path(p)):h for p,h in m.items()}
def subset(m,outer):
 for p,h in normalized(m).items():assert outer.get(p)==h,('missing/conflicting pin',p)
def refs(v):
 found={}
 if isinstance(v,dict):
  if isinstance(v.get('path'),str) and isinstance(v.get('sha256'),str):found[str(path(v['path']))]=v['sha256']
  for child in v.values():found.update(refs(child))
 elif isinstance(v,list):
  for child in v:found.update(refs(child))
 return found
plan,data,audit,manifest,tests=[load(OUT/(x+'.json'))for x in ('PLAN','DATA','AUDIT','MANIFEST','TESTS')]
pins=normalized(plan['source_hashes'])
for p,h in pins.items():assert sha(p)==h,('changed',p)
assert audit['source_hashes']==dict(pins,**{str(OUT/'PLAN.json'):sha(OUT/'PLAN.json')})
assert manifest=={'plan_sha256':sha(OUT/'PLAN.json'),'data_sha256':sha(OUT/'DATA.json'),'audit_sha256':sha(OUT/'AUDIT.json')}
assert audit['data_sha256']==manifest['data_sha256']and audit['plan_sha256']==manifest['plan_sha256']
assert tests['passed']is True and type(tests['tests'])is int and tests['tests']==21
assert tests['source_sha256']==sha('work/same_report145_union.py') and tests['test_sha256']==sha('work/test_same_report145_union.py')
assert tests['log']['sha256']==sha(tests['log']['path'])
known={
'aapl_amd':('outputs/monthly_same_report144_admission',20,16,4,'c40788c6487af79c11cd68cb762d16ccdb220b0e5c59001a9033923aa28adce9','bc310df597de8f5ec8ea433c11bb45cf09c8d4ce25e730282b141a41695a1e41'),
'msft_nvda':('outputs/monthly_msft_nvda_same_report_admission_v1',24,22,2,'d9778b1cbc2beb704b6e13ab41239bb3efa31730056a8a1ffdfc8a8d84925a46','1ebd4261b3d2cd8f1a417cdffdea5007a46f49b4cbaa2251b67b277f2c548f41')}
all_events=[];summaries={};event_digests={}
for name,(directory,n,accepted,rejected,dh,ah)in known.items():
 d,a,p,m=[load(directory+'/'+x+'.json')for x in ('DATA','AUDIT','PLAN','MANIFEST')]
 assert sha(directory+'/DATA.json')==dh and sha(directory+'/AUDIT.json')==ah
 assert a['data_sha256']==dh and a['plan_sha256']==sha(directory+'/PLAN.json') and a['policy_sha256']==sha(directory+'/POLICY.json')
 assert m['plan_sha256']==a['plan_sha256']and m['output_hashes']=={'DATA.json':dh,'AUDIT.json':ah}
 for mapping in ('source_hashes','source_code_hashes'):subset(a[mapping],pins);subset(p[mapping],pins)
 for filename in ('PLAN','POLICY','DATA','AUDIT','TESTS','MANIFEST'):assert pins[str(ROOT/directory/(filename+'.json'))]==sha(directory+'/'+filename+'.json')
 assert plan['packages'][name]==dict(data_path=directory+'/DATA.json',data_sha256=dh,audit_path=directory+'/AUDIT.json',audit_sha256=ah)
 assert len(d['events'])==n and sum(e['admitted']is True for e in d['events'])==accepted
 assert sum(e['admitted']is False for e in d['events'])==rejected
 assert data['assumptions'][name]==d['assumptions'] and data['missing_publications'][name]==d['missing_publications']
 subset(refs(d['events']),pins)
 all_events+=d['events']
 for e in d['events']:
  assert e['event_id']not in event_digests
  event_digests[e['event_id']]=hashlib.sha256(enc(e)).hexdigest()
 summaries[name]=dict(events=n,conditionally_admitted=accepted,rejected=rejected,raw_values=4*n,source_closure_count=len(a['source_hashes']))
 if name=='msft_nvda':
  b=load(directory+'/REVIEW_BINDING.json')
  assert a['review_binding_sha256']==sha(directory+'/REVIEW_BINDING.json')==m['review_binding_sha256']
  assert b['plan_sha256']==a['plan_sha256'] and b['stage']=='independent_review_binding'
  assert b['review_sha256']==sha(b['review_path'])==a['independent_review_sha256']=='5460f92bd8892dfa35514c54493d265d83c153d37030c92a7f23911973e73ae9'
  subset(b['source_hashes'],pins);subset(load(b['review_path'])['source_hashes'],pins)
  assert pins[str(ROOT/directory/'REVIEW_BINDING.json')]==a['review_binding_sha256']
assert data['events']==sorted(all_events,key=lambda e:(e['symbol'],e['available_from'],e['event_id']))
assert len({e['event_id']for e in data['events']})==44
assert {e['event_id']:hashlib.sha256(enc(e)).hexdigest()for e in data['events']}==event_digests
expected=dict(events=44,conditionally_admitted=38,rejected=6,raw_values=176)
assert data['counts']==expected and audit['counts']==expected
assert data['package_counts']=={n:{k:v for k,v in s.items()if k!='source_closure_count'}for n,s in summaries.items()}
for k in ('model_admitted','strategy_admitted','first_public_version_verified','complete_correction_history_verified','certified_daily_security_master'):assert data[k]is False
for k in ('financial_calculations','model_fits','ranks_computed','returns_computed','network_requests'):assert type(audit[k])is int and audit[k]==0
for k in ('ratios_computed','signs_computed','ranks_computed','returns_computed'):assert type(data[k])is int and data[k]==0
for p,h in pins.items():assert sha(p)==h,('changed while reviewing',p)
record=dict(schema_version=1,stage='independent_source_union_review',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 current_artifact_passed=True,source_only=True,counts=expected,packages=summaries,source_hashes_verified=len(pins),
 input_hashes={str((OUT/(x+'.json')).relative_to(ROOT)):sha(OUT/(x+'.json'))for x in ('PLAN','DATA','AUDIT','MANIFEST','TESTS')},
 implementation_hashes={p:sha(p)for p in ('work/same_report145_union.py','work/test_same_report145_union.py')},
 checks={k:True for k in ('all44_original_event_dictionaries_exact','all176_raw_values_preserved_by_dictionary_identity','all6_rejected_events_preserved','no_duplicate_event_or_issuer_package','availability_and_identity_unmodified','both_package_nested_source_closures_verified','new_review_binding_and_independent_review_verified','source_plan_policy_and_manifest_links_independently_verified','union_manifest_links_verified','all_declared_source_hashes_rehashed','no_financial_arithmetic_in_union_code')},
 event_sha256=event_digests,
 findings=[dict(id='registration_declared_closure_trust',severity='P2',scope='prospective_register_reuse',file='work/same_report145_union.py',lines=[104,108],
 description='Registration verifies hashes listed by the source audit but does not independently cross-check all embedded source PLAN/POLICY/MANIFEST/review-binding links or derive the required source closure. A newly supplied incomplete audit could therefore assert passing source checks while omitting provenance. The current fixed packages were separately validated by this review and have no missing pins or mismatched links.',
 current_artifact_affected=False,recommendation='Keep this frozen union unchanged. For any future registration require an independently approved exact package receipt or explicitly validate nested package links and required closure before accepting the audit.')],
 limitations=['This validates the current exact source union only; it does not certify general acceptance of arbitrary future source audits.','Original package research assumptions and MSFT accounting limitations are unchanged.','No archival first-public version, complete correction history, or daily security master certification.','No factor/model/strategy efficacy review.'],
 financial_calculations=0,model_fits=0,ranks_computed=0,returns_computed=0)
if REPORT.exists():
 previous=load(REPORT)
 assert {k:v for k,v in previous.items()if k!='created_utc'}=={k:v for k,v in record.items()if k!='created_utc'},'Existing review changed'
else:
 with REPORT.open('xb')as f:f.write((json.dumps(record,indent=2,sort_keys=True)+'\n').encode())
print(json.dumps(dict(current_artifact_passed=True,counts=expected,source_hashes_verified=len(pins),review_sha256=sha(REPORT))))
