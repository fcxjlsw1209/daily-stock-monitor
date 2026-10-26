from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,sys,re
from source_client import fetch
sys.path.insert(0,str(Path.cwd()/'work'))
from financial_html_tables import Tables
p=Path(__file__).resolve().parent;plan=json.loads((p/'PLAN.json').read_text());meta,f=fetch(plan['url']);doc=Tables();doc.feed(f.read_text());text=' '.join(' '.join(doc.visible).split());hits=[]
for pattern in [r'AT&T Inc\.[^\n]{0,120}',r'DALLAS.{0,100}',r'SOURCE AT&T']:
 hits.extend(m.group() for m in re.finditer(pattern,text))
r={'completed_utc':datetime.now(timezone.utc).isoformat(),'receipt':meta,'path':str(f.relative_to(p)),'identity_excerpt_candidates':hits,'semantic_admissions':0,'strategy_trials':0}
(p/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n');(p/'COMPLETION.json').write_text(json.dumps({'completed_utc':r['completed_utc'],'status':meta['status'],'semantic_admissions':0,'hashes':{str(q.relative_to(p)):hashlib.sha256(q.read_bytes()).hexdigest() for q in p.rglob('*') if q.is_file() and '__pycache__' not in q.parts}},indent=2)+'\n');print(json.dumps(r))
