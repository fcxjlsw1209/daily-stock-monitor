"""Cache exact public HTTP response and declared transfer decoding; no semantic certification."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from zoneinfo import ZoneInfo
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError
from urllib.parse import urlparse
import hashlib,json,gzip,subprocess,time
P=Path(__file__).resolve().parent
(P/'raw').mkdir(exist_ok=True)
ALLOWED={'www.prnewswire.com'}
def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,b):
 if p.exists():assert p.read_bytes()==b,str(p)
 else:
  with p.open('xb') as f:f.write(b)
def fetch(url):
 assert urlparse(url).scheme=='https' and urlparse(url).hostname in ALLOWED
 stem=sha(url.encode())[:20];rp=P/'raw'/f'{stem}.response';mp=P/'raw'/f'{stem}.meta.json';dp=P/'raw'/f'{stem}.decoded'
 if mp.exists():
  m=json.loads(mp.read_bytes());assert m['url']==url and m['status']==200 and urlparse(m['final_url']).hostname==urlparse(url).hostname and sha(rp.read_bytes())==m['response_sha256'] and sha(dp.read_bytes())==m['decoded_sha256'];return m,dp
 assert not (P/'STOPPED.json').exists(),'source stopped'
 time.sleep(0.6);m=dict(url=url,retrieved_at=datetime.now(timezone.utc).isoformat(),public_unauthenticated=True,request_count=1,accept_encoding='identity')
 try:
  with urlopen(Request(url,headers={'User-Agent':'LocalResearch/1.0','Accept-Encoding':'identity'}),timeout=30) as r:
   b=r.read(25*1024*1024+1);assert len(b)<=25*1024*1024
   m.update(status=r.status,final_url=r.geturl(),content_type=r.headers.get('Content-Type'),content_encoding=r.headers.get('Content-Encoding'),retry_after=r.headers.get('Retry-After'))
 except HTTPError as e:
  b=e.read(256*1024);m.update(status=e.code,final_url=e.geturl(),content_type=e.headers.get('Content-Type'),content_encoding=e.headers.get('Content-Encoding'),retry_after=e.headers.get('Retry-After'))
 except (URLError,TimeoutError) as e:
  b=b'';m.update(status=None,error_type=type(e).__name__)
 save(rp,b);m['response_sha256']=sha(b);m['response_bytes']=len(b)
 if m.get('status') in (403,429):
  local=datetime.now(ZoneInfo('America/Los_Angeles'));nextday=(local+timedelta(days=1)).replace(hour=0,minute=0,second=0,microsecond=0)
  save(P/'STOPPED.json',(json.dumps(dict(source=urlparse(url).hostname,status=m['status'],created_utc=m['retrieved_at'],retry_after=m.get('retry_after'),earliest_future_access_check_utc=nextday.astimezone(timezone.utc).isoformat(),instruction='No further requests to this source today; no alternate route or challenge bypass'),indent=2)+'\n').encode())
 enc=m.get('content_encoding')
 if enc=='br':
  v=subprocess.run(['node','--version'],capture_output=True,check=True,text=True).stdout.strip()
  d=subprocess.run(['node','-e','const fs=require("fs"),z=require("zlib");process.stdout.write(z.brotliDecompressSync(fs.readFileSync(0)));'],input=b,capture_output=True,check=True).stdout;m['decoder']='node:zlib.brotliDecompressSync';m['decoder_version']=v
 elif enc=='gzip':d=gzip.decompress(b);m['decoder']='python:gzip.decompress'
 else:assert enc in (None,'identity','');d=b;m['decoder']='identity'
 m.update(decoded_sha256=sha(d),decoded_bytes=len(d),response_path=str(rp.relative_to(P)),decoded_path=str(dp.relative_to(P)))
 save(dp,d);save(mp,(json.dumps(m,sort_keys=True,indent=2)+'\n').encode())
 assert m.get('status')==200 and urlparse(m.get('final_url','')).hostname==urlparse(url).hostname,f'source failed {m.get("status")}'
 return m,dp
