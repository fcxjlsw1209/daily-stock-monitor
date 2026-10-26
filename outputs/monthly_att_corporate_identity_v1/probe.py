from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
from source_client import fetch
p=Path(__file__).resolve().parent;plan=json.loads((p/'PLAN.json').read_text());result={}
try:
 m,f=fetch(plan['url']);result={'receipt':m,'path':str(f.relative_to(p))};print(m['status'],m['response_bytes'],flush=True)
except Exception as e:result={'error':str(e),'error_type':type(e).__name__};print(result,flush=True)
(p/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
(p/'SOURCE_COMPLETION.json').write_text(json.dumps({'completed_utc':datetime.now(timezone.utc).isoformat(),'status':'SOURCE_OK' if 'receipt' in result else 'SOURCE_FAILED','semantic_admissions':0,'strategy_trials':0,'artifact_hashes':{str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file() and '__pycache__' not in f.parts}},indent=2)+'\n')
