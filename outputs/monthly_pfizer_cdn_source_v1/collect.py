from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
from source_client import fetch
p=Path(__file__).resolve().parent
plan=json.loads((p/'PLAN.json').read_text())
r={'completed_utc':datetime.now(timezone.utc).isoformat(),'semantic_admissions':0,'strategy_trials':0}
try:
 receipt,path=fetch(plan['url'])
 r.update(receipt=receipt,pdf_signature=path.read_bytes().startswith(b'%PDF-'),file=str(path.relative_to(p)))
except Exception as e:r['error']=str(e)
(p/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
print({k:v for k,v in r.items() if k!='receipt'})
