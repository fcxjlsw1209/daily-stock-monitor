"""One-shot public-source probe for the missing 2019-04-29 A-share quote."""
import hashlib,json,sys,urllib.request,urllib.error
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_ptr_pool_source169/missing_quote'
SOURCES={
 'sina_2019q2.html':'https://vip.stock.finance.sina.com.cn/corp/go.php/vMS_MarketHistory/stockid/601857.phtml?year=2019&jidu=2',
 'cjsc_20190430.pdf':'https://cjsc.com.hk/sites/default/files/research/%E6%B8%AF%E8%82%A1%E6%AF%8F%E6%97%A5%E8%81%9A%E7%84%A6-20190430.pdf',
}
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def register():
 OUT.mkdir(exist_ok=True)
 write(OUT/'PLAN.json',dict(registered_utc=now(),code_sha256=sha(Path(__file__)),sources=SOURCES,
  quote_audit_sha256=sha(ROOT/'outputs/monthly_ptr_pool_source169/QUOTE_AUDIT.json'),
  scope='Find independent daily quote for 601857.SH on2019-04-29 only; do not fill source, compute caps or returns.',
  discovery='Sina historical daily table URL pattern; public search found CJSC20190430 daily publication with A/H price table.',
  limits=['CJSC is a public broker report citing Wind, not exchange direct quote; report date Apr30 is after target asofApr29. Do not assume historical availability before publication.',
   'Web extraction suggests A7.40 and H5.06 but local body, visual table alignment and quote date still require verification.',
   'Sina web tool gave inaccessible internal error without HTTP code; this is first direct acquisition probe.',
   'Single request per source; no automatic retries or bypass;403/429stop provider forUTCday.']))
 print('Registered missing-quote sources.')
def collect():
 p=read(OUT/'PLAN.json');assert p['code_sha256']==sha(Path(__file__))
 for name,url in p['sources'].items():
  dest=OUT/name; receipt=OUT/(name+'.receipt.json')
  if receipt.exists():
   r=read(receipt)
   if r.get('body_sha256'):assert sha(dest)==r['body_sha256']
   continue
  r=dict(url=url,requested_utc=now(),plan_sha256=sha(OUT/'PLAN.json'))
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 public historical research'}),timeout=40) as z:
    body=z.read();r.update(status=z.status,final_url=z.url,content_type=z.headers.get('Content-Type'),retry_after=z.headers.get('Retry-After'))
  except urllib.error.HTTPError as e:
   body=e.read();r.update(status=e.code,final_url=e.url,content_type=e.headers.get('Content-Type'),retry_after=e.headers.get('Retry-After'))
  except Exception as e:body=None;r.update(status='transport_error',error=type(e).__name__,message=str(e))
  if body is not None:
   with dest.open('xb') as f:f.write(body)
   r.update(body_sha256=sha(dest),bytes=len(body))
  r['finished_utc']=now();write(receipt,r)
  print(json.dumps(dict(name=name,status=r['status'],bytes=r.get('bytes'))),flush=True)
if __name__=='__main__':{'register':register,'collect':collect}[sys.argv[1]]()
