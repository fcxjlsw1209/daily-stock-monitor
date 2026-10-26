"""Public issuer archival evidence, cached by URL+SHA; no price or trading work."""
import datetime as dt,hashlib,json,re,time,urllib.request,urllib.error,urllib.parse,subprocess
from pathlib import Path
from functools import lru_cache
from bs4 import BeautifulSoup
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];CUTOFF='2026-09-24'; START='2011-01-01'
PDF_PYTHON='/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(b):return hashlib.sha256(b).hexdigest()
def rel(p):return str(p.relative_to(ROOT))
def fetch(url,ticker,stem=None):
 host=urllib.parse.urlsplit(url).netloc
 if host.endswith(('sec.gov','finance.yahoo.com','alphavantage.co')):raise RuntimeError('Globally stopped vendor or SEC direct source')
 suffix=Path(urllib.parse.urlsplit(url).path).suffix
 if suffix not in ('.pdf','.xlsx','.html','.js','.json','.txt'):suffix='.html'
 raw=HERE/ticker/'raw';raw.mkdir(parents=True,exist_ok=True)
 p=raw/((stem or sha(url.encode())[:20])+suffix);mp=p.with_name(p.name+'.meta.json')
 if p.exists() and mp.exists():
  m=json.loads(mp.read_text());b=p.read_bytes()
  if m['url']!=url or m['sha256']!=sha(b):raise ValueError('Cache URL/hash mismatch')
  if m['status']!=200:raise RuntimeError('Cached HTTP '+str(m['status']))
  return b,p,m
 stops=HERE/'SOURCE_STOPS.json'; stopped=json.loads(stops.read_text()) if stops.exists() else {}
 if host in stopped:raise RuntimeError('Stopped source '+host)
 request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (public archival research)'})
 try:
  with urllib.request.urlopen(request,timeout=45) as r:b=r.read();m=dict(url=url,final_url=r.geturl(),status=r.status,content_type=r.headers.get('Content-Type'))
 except urllib.error.HTTPError as e:b=e.read();m=dict(url=url,status=e.code,retry_after=e.headers.get('Retry-After'))
 except (urllib.error.URLError,TimeoutError) as e:
  dump(mp,dict(url=url,status=None,retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat(),sha256=None,bytes=None,error=str(e),response_bytes_available=False,public_unauthenticated=True));raise
 m.update(retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat(),sha256=sha(b),bytes=len(b),public_unauthenticated=True)
 p.write_bytes(b);dump(mp,m)
 if m['status'] in (403,429):
  stopped=json.loads(stops.read_text()) if stops.exists() else {};stopped[host]=dict(m,policy='Stop this host for this task, no bypass');dump(stops,stopped);print('SOURCE_STOP',host,m['status'],m.get('retry_after'),flush=True)
 if m['status']!=200:raise RuntimeError('HTTP '+str(m['status'])+' '+url)
 time.sleep(.15)
 return b,p,m

def text(b):
 s=BeautifulSoup(b,'html.parser')
 for x in s(['style','script']):x.decompose()
 return clean(s.get_text(' ',strip=True))
def clean(t):return re.sub(r'\s+',' ',t.replace('\u00a0',' ').replace('–','-').replace('—','-'))
def proof(p,url,excerpt=None):
 r=dict(evidence_path=rel(p),sha256=sha(p.read_bytes()),source_url=url)
 if excerpt:r['excerpt']=excerpt
 return r
@lru_cache(maxsize=1024)
def pdf_text(p):
 target=p.with_suffix('.txt');meta=p.with_name(p.name+'.textmeta.json');h=sha(p.read_bytes())
 if target.exists() and meta.exists():
  m=json.loads(meta.read_text())
  if m['source_sha256']==h and m['text_sha256']==sha(target.read_bytes()):return target.read_text()
 code='import sys; from pypdf import PdfReader; print("\\n".join(p.extract_text() or "" for p in PdfReader(sys.argv[1]).pages))'
 txt=subprocess.check_output([PDF_PYTHON,'-c',code,str(p)],text=True)
 target.write_text(txt);dump(meta,dict(source_sha256=h,text_sha256=sha(target.read_bytes()),parser='bundled pypdf'))
 return txt

@lru_cache(maxsize=128)
def workbook_text(p):
 """Extract only original header cells, without recalculating or changing workbooks."""
 target=p.with_name(p.name+'.headers.json');meta=p.with_name(p.name+'.headertextmeta.json');h=sha(p.read_bytes())
 if target.exists() and meta.exists():
  m=json.loads(meta.read_text())
  if m['source_sha256']==h and m['text_sha256']==sha(target.read_bytes()):return clean(' '.join(x['sheet']+' '+x['text']for x in json.loads(target.read_text())))
 result=subprocess.check_output([PDF_PYTHON,str(HERE/'read_workbook_headers.py'),str(p)],text=True);target.write_text(result);dump(meta,dict(source_sha256=h,text_sha256=sha(target.read_bytes()),parser='bundled openpyxl read-only data-only original header cells, no edits or recalculation'))
 return clean(' '.join(x['sheet']+' '+x['text']for x in json.loads(result)))
