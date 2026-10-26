"""Task-scoped official-source cache. No SEC/Yahoo/AV requests or trading writes."""
import datetime as dt, hashlib,json,re,time,urllib.request,urllib.error,urllib.parse
from pathlib import Path
from bs4 import BeautifulSoup
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
CUTOFF='2026-09-24'
def dump(p,o):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(b):return hashlib.sha256(b).hexdigest()
def rel(p):return str(p.relative_to(ROOT))
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def text(b):
 s=BeautifulSoup(b,'html.parser')
 for x in s(['script','style']):x.decompose()
 return re.sub(r'\s+',' ',s.get_text(' ',strip=True)).replace('\xa0',' ')
def fetch(ticker,url,name=None):
 host=urllib.parse.urlsplit(url).netloc
 if any(h in host for h in ['sec.gov','yahoo.','alphavantage']):raise ValueError('prohibited paused source')
 d=HERE/ticker/'raw';d.mkdir(exist_ok=True,parents=True)
 stem=name or sha(url.encode())[:20];p=d/(stem+'.bin');mp=d/(stem+'.meta.json')
 if p.exists() and mp.exists():
  m=json.loads(mp.read_text());b=p.read_bytes()
  assert m['url']==url and m['sha256']==sha(b)
  if m['status']!=200:raise RuntimeError('Cached failed '+str(m['status']))
  return b,p,m
 stops=HERE/'SOURCE_STOPS.json';j=json.loads(stops.read_text()) if stops.exists() else {}
 if host in j:raise RuntimeError('Stopped source '+host)
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (public earnings research)'}),timeout=45) as r:
   b=r.read();m=dict(url=url,final_url=r.geturl(),status=r.status,content_type=r.headers.get('Content-Type'))
 except urllib.error.HTTPError as e:b=e.read();m=dict(url=url,status=e.code,retry_after=e.headers.get('Retry-After'))
 except Exception as e:
  dump(d/(stem+'.error.json'),dict(url=url,retrieved_at=now(),error=str(e)));raise
 m.update(retrieved_at=now(),sha256=sha(b),bytes=len(b),public_unauthenticated=True)
 p.write_bytes(b);dump(mp,m)
 if m['status'] in (403,429):
  j[host]=dict(m,policy='Stopped through the current America/Los_Angeles day and until explicit manual resumption; UTC rollover never resumes automatically');dump(stops,j)
  print('SOURCE_STOP',host,m['status'],m.get('retry_after'),flush=True)
 if m['status']!=200:raise RuntimeError('HTTP '+str(m['status'])+' '+url)
 return b,p,m
