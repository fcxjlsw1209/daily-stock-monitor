"""Auditable official-source batch. No vendor requests, prices or trading.

Downloads are cached and SHA checked. 403/429 stop the entire direct host for
the local research day; recorded stops are retained across UTC midnight. Build is offline; events require separately evidenced release and
actual fiscal-period dates. An archive calendar alone is never publication.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import threading
import urllib.request
import urllib.error
from urllib.parse import urlparse, urljoin, quote
from bs4 import BeautifulSoup

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RAW=HERE/'raw'
RAW.mkdir(exist_ok=True)
WRITE_LOCK=threading.Lock()
TICKERS='XOM UNH BABA CVX PFE MRK ASML LLY ABBV TM NVS SHEL NVO SAP AZN CHL ORCL GE BA DHR'.split()
CUTOFF='2026-09-24'
PDF_PYTHON='/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
ZONES={t:'America/New_York' for t in TICKERS}
ZONES.update(XOM='America/Chicago',UNH='America/Chicago',CVX='America/Los_Angeles',LLY='America/Indiana/Indianapolis',ABBV='America/Chicago',BABA='Asia/Shanghai',TM='Asia/Tokyo',ASML='Europe/Amsterdam',NVS='Europe/Zurich',SHEL='Europe/London',NVO='Europe/Copenhagen',SAP='Europe/Berlin',AZN='Europe/London',CHL='Asia/Hong_Kong',ORCL='America/Los_Angeles',BA='America/Chicago')
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def rel(p):return str(Path(p).relative_to(ROOT))
def clean(t):return re.sub(r'\s+',' ',t.replace('\u00a0',' ').replace('–','-').replace('—','-'))
def load(p,default):return json.loads(Path(p).read_text()) if Path(p).exists() else default
def fetch(url,ticker,role='discovery',data=None):
 host=urlparse(url).hostname;today=utc()[:10]
 key=sha(url.encode()+(json.dumps(data,sort_keys=True).encode() if data is not None else b''))[:20];folder=RAW/ticker;folder.mkdir(exist_ok=True)
 p=folder/(key+'.bin');m=folder/(key+'.meta.json')
 if p.exists() and m.exists():
  info=load(m,{})
  if info['url']!=url or info['sha256']!=sha(p.read_bytes()):raise ValueError('cache checksum/source mismatch')
  return p,info
 stops=load(HERE/'SOURCE_STOPS.json',[])
 if host in ('www.sec.gov','sec.gov','data.sec.gov','query1.finance.yahoo.com','query2.finance.yahoo.com','www.alphavantage.co') or any(x['host']==host for x in stops):raise RuntimeError('source stopped: '+host)
 failures=load(HERE/'DOWNLOAD_FAILURES.json',[])
 if any(x['url']==url and x['utc_date']==today and x.get('http_status') is not None for x in failures):raise RuntimeError('cached HTTP failure: '+url)
 try:
  transport_url=quote(url,safe="/:?#[]@!$&'()*+,;=%")
  with urllib.request.urlopen(urllib.request.Request(transport_url,data=json.dumps(data).encode() if data is not None else None,headers={'User-Agent':'Mozilla/5.0 (Historical financial releases research)','Content-Type':'application/json'}),timeout=35) as response:
   body=response.read();info=dict(url=url,final_url=response.geturl(),status=response.status,retrieved_at=utc(),sha256=sha(body),bytes=len(body),content_type=response.headers.get('Content-Type'),role=role,ticker=ticker,source_representation='original_http_bytes',request_method='POST' if data is not None else 'GET',request_payload=data)
 except Exception as e:
  failure=dict(url=url,host=host,ticker=ticker,role=role,utc_date=today,retrieved_at=utc(),error=str(e),http_status=getattr(e,'code',None),retry_after=e.headers.get('Retry-After') if isinstance(e,urllib.error.HTTPError) else None)
  with WRITE_LOCK:
   failures=load(HERE/'DOWNLOAD_FAILURES.json',[]);failures.append(failure);dump(HERE/'DOWNLOAD_FAILURES.json',failures)
   if failure['http_status'] in (403,429):
    stops=load(HERE/'SOURCE_STOPS.json',[]);stops.append(failure);dump(HERE/'SOURCE_STOPS.json',stops)
  raise
 p.write_bytes(body);dump(m,info);time.sleep(.1)
 return p,info
def text_of(p):
 p=Path(p);b=p.read_bytes();dest=p.with_suffix('.txt');meta=p.with_suffix('.text.json')
 if dest.exists() and meta.exists():
  d=load(meta,{})
  if d.get('source_sha256')==sha(b) and d.get('text_sha256')==sha(dest.read_bytes()):return dest.read_text()
 if b.startswith(b'%PDF'):
  code='import sys;from pypdf import PdfReader;print("\\n".join(p.extract_text() or "" for p in PdfReader(sys.argv[1]).pages))'
  t=subprocess.check_output([PDF_PYTHON,'-c',code,str(p)],text=True,stderr=subprocess.DEVNULL);parser='pypdf'
 else:
  s=BeautifulSoup(b,'html.parser')
  for n in s(['script','style']):n.decompose()
  t=s.get_text(' ',strip=True);parser='BeautifulSoup HTML text'
 dest.write_text(t);dump(meta,dict(source_sha256=sha(b),text_sha256=sha(dest.read_bytes()),parser=parser));return t
def links(p,url):
 s=BeautifulSoup(Path(p).read_bytes(),'html.parser')
 return [dict(url=urljoin(url,a.get('href','')),text=a.get_text(' ',strip=True)) for a in s.find_all('a',href=True)]
def get(url,ticker,role='discovery'):
 try:
  p,m=fetch(url,ticker,role);t=text_of(p)
  print(json.dumps(dict(ticker=ticker,url=url,path=str(p),bytes=m['bytes'],preview=clean(t)[:700],links=links(p,url) if not p.read_bytes().startswith(b'%PDF') else [])),flush=True)
 except Exception as e:print(json.dumps(dict(ticker=ticker,url=url,error=str(e))),flush=True)

if __name__=='__main__':
 if len(sys.argv)>2:get(sys.argv[2],sys.argv[1])
