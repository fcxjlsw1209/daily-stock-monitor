from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
from source_client import fetch
p=Path(__file__).resolve().parent;plan=json.loads((p/'PLAN.json').read_text());docs=[];errors=[]
for a in plan['queue']:
 try:
  m,f=fetch(a['request_url']);docs.append({**a,'receipt':m,'path':str(f.relative_to(p))});print(a['report_end'],m['status'],m['response_bytes'],flush=True)
 except Exception as e:errors.append({'report_end':a['report_end'],'error':str(e)});break
(p/'RESULT.json').write_text(json.dumps({'documents':docs,'errors':errors},indent=2)+'\n')
(p/'SOURCE_COMPLETION.json').write_text(json.dumps({'completed_utc':datetime.now(timezone.utc).isoformat(),'status':'COMPLETE' if not errors else 'PARTIAL_STOPPED','documents':len(docs),'errors':errors,'semantic_admissions':0,'strategy_trials':0,'hashes':{str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file() and '__pycache__' not in f.parts}},indent=2)+'\n')
print('documents',len(docs),'errors',errors,flush=True)
