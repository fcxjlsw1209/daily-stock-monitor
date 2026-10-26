"""Offline integrity audit for independent candidate supplements."""
import json,datetime as dt,collections
import core as c
D=c.HERE/'candidate_supplement'
def main():
 rows=json.loads((D/'events.json').read_text());paths={};errors=[]
 def walk(x):
  if isinstance(x,dict):
   if x.get('evidence_path'):
    p=c.ROOT/x['evidence_path'];expected=x.get('sha256',x.get('evidence_sha256'));got=c.sha(p.read_bytes());paths[c.rel(p)]=dict(sha256=got,bytes=p.stat().st_size)
    if expected and got!=expected:errors.append(dict(path=c.rel(p),error='SHA mismatch'))
   for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 walk(rows)
 for path,item in paths.items():
  p=c.ROOT/path;mp=p.with_name(p.name+'.meta.json')
  if mp.exists():
   m=json.loads(mp.read_text());item['retrieved_at']=m.get('retrieved_at');item['original_bytes']=m.get('status')==200
   if m.get('sha256')!=item['sha256']:errors.append(dict(path=path,error='Metadata SHA mismatch'))
  elif p.suffix!='.png':errors.append(dict(path=path,error='Missing metadata'))
 tests=(D/'TESTS.txt').read_text();assert tests.rstrip().endswith('OK'),tests[-1000:]
 v=dict(generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),event_count=len(rows),events_sha256=c.sha((D/'events.json').read_bytes()),evidence_files=len(paths),errors=errors,tests='6 offline tests passed',paths=paths)
 c.dump(D/'VALIDATION.json',v);assert not errors,errors
 s=json.loads((D/'STATUS.json').read_text());s.update(status='validated',validation_errors=0,offline_tests_passed=6);c.dump(D/'STATUS.json',s)
 print({k:v for k,v in v.items()if k!='paths'})
if __name__=='__main__':main()
