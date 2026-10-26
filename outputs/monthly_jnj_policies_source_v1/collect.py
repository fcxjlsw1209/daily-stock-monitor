from pathlib import Path
from datetime import datetime,timezone
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import json,hashlib
from source_client import fetch
p=Path(__file__).resolve().parent;plan=json.loads((p/'PLAN.json').read_text())
r={'created_utc':datetime.now(timezone.utc).isoformat(),'semantic_admissions':0,'strategy_trials':0}
try:
 m,path=fetch(plan['url']);soup=BeautifulSoup(path.read_bytes(),'html.parser')
 text=soup.get_text(' ',strip=True);links=[{'url':urljoin(plan['url'],a['href']),'label':a.get_text(' ',strip=True)}for a in soup.find_all('a',href=True)]
 r.update(receipt=m,body_bytes=len(path.read_bytes()),maintenance='site maintenance'in text.lower(),report_links=[x for x in links if 'annual report'in x['label'].lower()])
 r['valid_report_archive']=not r['maintenance'] and len(r['report_links'])>0
except Exception as e:r['error']=str(e)
(p/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
(p/'COMPLETION.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'files':{str(q.relative_to(p)):hashlib.sha256(q.read_bytes()).hexdigest()for q in p.rglob('*')if q.is_file() and '__pycache__'not in str(q)},'semantic_admissions':0,'strategy_trials':0},indent=2)+'\n')
print(r)
