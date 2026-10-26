from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
from source_client import fetch
p=Path(__file__).resolve().parent
assert not (p/'RESULT.json').exists()
plan=json.loads((p/'PLAN.json').read_text()); rows=[];errors=[]
for url in plan['urls']:
 try:
  m,f=fetch(url);assert b'<html' in f.read_bytes().lower() and b'Visa' in f.read_bytes();rows.append({'url':url,'receipt':m}); print('downloaded',url,flush=True)
 except Exception as e:errors.append({'url':url,'error':str(e)});break
(p/'RESULT.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'documents':rows,'errors':errors,'semantic_admissions':0,'strategy_trials':0},indent=2)+'\n')
(p/'COMPLETION.json').write_text(json.dumps({'files':{str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest()for f in p.rglob('*')if f.is_file()and '__pycache__'not in str(f)},'documents':len(rows),'errors':errors,'semantic_admissions':0,'strategy_trials':0},indent=2)+'\n')
print('complete',len(rows),errors)
