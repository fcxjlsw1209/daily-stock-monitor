from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urljoin
from datetime import datetime,timezone
import json,hashlib
from source_client import fetch
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,t,attrs):
  if t=='a':
   a=dict(attrs)
   if a.get('href'):self.links.append(a['href'])
p=Path(__file__).resolve().parent;plan=json.loads((p/'PLAN.json').read_text());result={}
try:
 m,f=fetch(plan['seed_url']);html=Links();html.feed(f.read_text());links=sorted({urljoin(m['final_url'],u) for u in html.links if '/news-releases/' in u and ('facebook-reports-' in u or 'meta-reports-' in u)});result={'receipt':m,'links':links,'path':str(f.relative_to(p))};print('links',len(links));print('\n'.join(links))
except Exception as e:result={'error':str(e),'error_type':type(e).__name__};print(result)
(p/'DISCOVERY.json').write_text(json.dumps(result,indent=2)+'\n')
(p/'DISCOVERY_COMPLETION.json').write_text(json.dumps({'completed_utc':datetime.now(timezone.utc).isoformat(),'semantic_admissions':0,'hashes':{str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file() and f.name!='DISCOVERY_COMPLETION.json' and '__pycache__' not in f.parts}},indent=2)+'\n')
