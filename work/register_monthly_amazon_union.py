"""Register a count-only union of already certified data, never a strategy trial."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent.parent
P=ROOT/'outputs/monthly_sec_fullpool_preflight_v6'
BASE='outputs/monthly_sec_fullpool_preflight_v5/'
ADDED='outputs/monthly_sec_amazon_original_admission_v2/'
def read(p):return json.loads((ROOT/p).read_bytes())
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def blob(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def create(p,data):
 if p.exists():assert p.read_bytes()==data,str(p)
 else:
  with p.open('xb') as f:f.write(data)
hashes={}
def bind(p,h=None):
 actual=sha(p);assert h is None or actual==h,p
 if p in hashes:assert hashes[p]==actual
 hashes[p]=actual
for folder in (BASE,ADDED):
 d=read(folder+'DESIGN.json');c=read(folder+'COMPLETION.json');plan=read(folder+'PLAN.json')
 assert c['independent_review']['passed'] and c['strategy_trials']==0
 for mapping in (d['source_hashes'],plan['implementation_hashes'],plan['independent_review_hashes']):
  for path,h in mapping.items():bind(path,h)
 for n in ('DESIGN.json','PLAN.json','COMPLETION.json'):bind(folder+n)
 for n,h in c['artifact_hashes'].items():bind(n if n.startswith(('work/','outputs/')) else folder+n,h)
 review=c['independent_review'];bind(review['path'],review['sha256']);r=read(review['path'])
 assert r['passed'] and not r['errors']
 for key in ('source_hashes','implementation_hashes','independent_review_hashes','auditor_source_hashes','artifact_hashes'):
  for path,h in r.get(key,{}).items():bind(path,h)
 if 'schema_adapter' in r:
  for path,h in r['schema_adapter']['source_hashes'].items():bind(path,h)
  e=r['schema_adapter']['erratum'];bind(e['path'],e['sha256'])
for n in ('preflight.py',):bind('outputs/monthly_sec_fullpool_preflight_v1/'+n)
design=dict(schema_version=1,stage='count_only_coverage',policy_id='monthly_sec_fullpool_semantic_coverage_v6',
 registered_at_utc=datetime.now(timezone.utc).isoformat(),source_hashes=hashes,
 method='Replay frozen v5 coverage and Amazon certified sidecar, merge disjoint exact source keys and all required pair IDs; retain provenance class and unchanged original flags; recount with frozen breadth function.',
 expected=dict(source_records=3904,signal_dates=122,baseline_rows=365,additional_rows=53,combined_rows=418,combined_unique_pairs=193),
 thresholds=read(BASE+'DESIGN.json')['thresholds'],
 contract=dict(factor_ready=False,strategy_trials=0,new_semantic_admissions=0,network_requests=0,observer_files_changed=False,
  source_flags_unchanged=True,raw_alphabet_facts_provenance_verified=False,raw_amazon_facts_provenance_verified=True,growth_factor_rank_order_return_computed=False))
create(P/'DESIGN.json',blob(design));h=sha(str((P/'DESIGN.json').relative_to(ROOT)))
build=(ROOT/BASE/'build.py').read_text().replace("BASE = 'outputs/monthly_sec_fullpool_preflight_v4/'", "BASE = 'outputs/monthly_sec_fullpool_preflight_v5/'").replace("ADDED = 'outputs/monthly_sec_alphabet_archive_admission_v1/'", "ADDED = 'outputs/monthly_sec_amazon_original_admission_v2/'").replace('e120364bf590f057ea000953dd5ad5cb215bb8e27dbf0e9a026c9ee7dda27a10',h)
create(P/'build.py',build.encode());print(json.dumps(dict(design_sha256=h,sources=len(hashes))))
