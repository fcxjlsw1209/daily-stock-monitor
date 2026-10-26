"""Validate all new-event and explanatory original evidence without modifying sources."""
import json,datetime as dt,re
import core as c
from build_structural_review import D,FROZEN

def main():
 paths={};errors=[]
 def walk(x):
  if isinstance(x,dict):
   if x.get('evidence_path'):
    p=c.ROOT/x['evidence_path'];got=c.sha(p.read_bytes());expected=x.get('sha256',x.get('evidence_sha256'));path=c.rel(p)
    if expected and got!=expected:errors.append(dict(path=path,error='SHA mismatch'))
    metas=[p.with_name(p.name+'.meta.json'),p.with_suffix('.meta.json'),p.with_suffix('.json')];mp=next((m for m in metas if m.exists()),None)
    if mp is None:errors.append(dict(path=path,error='Missing original response metadata'))
    else:
     m=json.loads(mp.read_text());status=m.get('status',m.get('http_status'));retrieved=m.get('retrieved_at');url=m.get('url');paths[path]=dict(sha256=got,bytes=p.stat().st_size,metadata_path=c.rel(mp),metadata_sha256=c.sha(mp.read_bytes()),retrieved_at=retrieved,source_url=url,http_status=status)
     if m.get('sha256')!=got or status!=200 or not retrieved:errors.append(dict(path=path,error='Original metadata invalid'))
     if x.get('source_url') and x['source_url']!=url:errors.append(dict(path=path,error='Metadata URL mismatch'))
   for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 rows=json.loads((D/'events.json').read_text());review=json.loads((D/'DISPOSITIONS.json').read_text());walk(rows);walk(review['dispositions'])
 for name,h in FROZEN.items():
  if c.sha((c.HERE/name).read_bytes())!=h:errors.append(dict(path=name,error='Frozen snapshot modified'))
 tests=(D/'TESTS.txt').read_text();assert tests.rstrip().endswith('OK');nt=int(re.search(r'Ran (\d+) tests',tests).group(1));assert nt==9
 unused=[]
 for p in sorted((D/'indexed').glob('*.json')):
  if p.name.endswith('.meta.json'):continue
  m=json.loads(p.with_name(p.name+'.meta.json').read_text());assert m['sha256']==c.sha(p.read_bytes());unused.append(dict(path=c.rel(p),sha256=m['sha256'],retrieved_at=m['retrieved_at'],reason='Exploratory public-index response; not used for admission or original-source facts.'))
 v=dict(generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),event_count=len(rows),events_sha256=c.sha((D/'events.json').read_bytes()),dispositions_sha256=c.sha((D/'DISPOSITIONS.json').read_bytes()),dispositions=len(review['dispositions']),candidate_observations=sum(len(r['signal_dates'])*len(r['scopes'])for r in review['dispositions']),original_evidence_files=len(paths),evidence=paths,unused_indexed_files=unused,errors=errors,offline_tests_passed=nt,frozen_snapshots_unchanged=not errors)
 c.dump(D/'VALIDATION.json',v);assert not errors,errors
 status=json.loads((D/'STATUS.json').read_text());status.update(status='validated',validation_errors=0,offline_tests_passed=nt,original_evidence_files=len(paths),candidate_observations=v['candidate_observations']);c.dump(D/'STATUS.json',status)
 print({k:v for k,v in v.items()if k not in ('evidence','unused_indexed_files')})
if __name__=='__main__':main()
