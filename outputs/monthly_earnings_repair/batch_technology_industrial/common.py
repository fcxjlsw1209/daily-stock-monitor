"""Task-scoped official-source cache. No SEC/Yahoo/AV requests or trading writes."""
import datetime as dt, hashlib,json,re,time,urllib.request,urllib.error,urllib.parse
from pathlib import Path
from bs4 import BeautifulSoup
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
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
CONFIG={
 'TSLA':('Tesla','https://ir.tesla.com','America/Los_Angeles'),
 'INTC':('Intel','https://www.intc.com','America/Los_Angeles'),
 'CSCO':('Cisco','https://investor.cisco.com','America/Los_Angeles'),
 'AVGO':('Broadcom','https://investors.broadcom.com','America/Los_Angeles'),
 'NFLX':('Netflix','https://ir.netflix.net','America/Los_Angeles'),
 'ADBE':('Adobe','https://www.adobe.com/investor-relations.html','America/Los_Angeles'),
 'CRM':('Salesforce','https://investor.salesforce.com','America/Los_Angeles'),
 'AMD':('Advanced Micro Devices','https://ir.amd.com','America/Los_Angeles'),
 'PLTR':('Palantir','https://investors.palantir.com','America/Denver'),
 'MU':('Micron','https://investors.micron.com','America/Boise'),
 'TMUS':('T-Mobile US','https://investor.t-mobile.com','America/Los_Angeles'),
 'CAT':('Caterpillar','https://investors.caterpillar.com','America/Chicago'),
 'LRCX':('Lam Research','https://investor.lamresearch.com','America/Los_Angeles'),
 'SPCX':('Space Exploration Technologies','https://ir.spacex.com','America/Chicago'),
 'AMAT':('Applied Materials','https://ir.appliedmaterials.com','America/Los_Angeles'),
 'SKHY':('SK hynix','https://news.skhynix.com','Asia/Seoul'),
 'ACN':('Accenture','https://investor.accenture.com','America/New_York'),
 'ARM':('Arm Holdings','https://investors.arm.com','Europe/London'),
 'KLAC':('KLA','https://ir.kla.com','America/Los_Angeles'),
 'IBM':('IBM','https://www.ibm.com/investor','America/New_York'),
 'HPQ':('HP','https://investor.hp.com','America/Los_Angeles'),
 'QCOM':('Qualcomm','https://investor.qualcomm.com','America/Los_Angeles'),
 'PDD':('PDD Holdings','https://investor.pddholdings.com','Asia/Shanghai'),
}
if __name__=='__main__':
 for k,(name,url,tz) in CONFIG.items():
  try:
   b,p,m=fetch(k,url,'home');print(k,len(b),m['final_url'],flush=True)
  except Exception as e:print(k,'ERROR',str(e),flush=True)
