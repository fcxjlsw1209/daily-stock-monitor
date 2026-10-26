"""One unadjusted historical OHLC probe, separate from offline validation."""
import hashlib
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_ptr_pool_source169/missing_quote'
URL='https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh601857,day,2019-04-22,2019-04-30,20,'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def register():
    write(OUT/'TENCENT_PLAN.json',dict(created_utc=now(),url=URL,code_sha256=sha(Path(__file__)),
       source='Tencent public daily candlestick endpoint; blank adjustment parameter requests unadjusted bars.',
       purpose='Independent OHLC check for2019-04-29 and nearby Yahoo overlap only; no valuation.',
       policy='One request; reuse any receipt;403/429stop; response may be unavailable or revised; HTTP200 not sufficent.',
       quotation='Instrument sh601857; day; requested2019-04-22..2019-04-30; count20; no adjustment',
       prior_extraction_sha256=sha(OUT/'EXTRACTION.json')))
def collect():
    p=read(OUT/'TENCENT_PLAN.json');assert p['code_sha256']==sha(Path(__file__))
    dest=OUT/'tencent_201904.json';receipt=OUT/'tencent_201904.receipt.json'
    if receipt.exists():
        old=read(receipt)
        if old.get('body_sha256'):assert sha(dest)==old['body_sha256']
        print(json.dumps(old));return
    r=dict(url=p['url'],requested_utc=now(),plan_sha256=sha(OUT/'TENCENT_PLAN.json'))
    try:
        req=urllib.request.Request(p['url'],headers={'User-Agent':'Mozilla/5.0 public historical research'})
        with urllib.request.urlopen(req,timeout=40) as response:
            data=response.read();r.update(status=response.status,final_url=response.url,
                content_type=response.headers.get('Content-Type'),retry_after=response.headers.get('Retry-After'))
    except urllib.error.HTTPError as e:
        data=e.read();r.update(status=e.code,final_url=e.url,retry_after=e.headers.get('Retry-After'))
    except Exception as e:
        data=None;r.update(status='transport_error',error=type(e).__name__,detail=str(e))
    if data is not None:
        with dest.open('xb') as f:f.write(data)
        r.update(bytes=len(data),body_sha256=sha(dest))
    r['finished_utc']=now();write(receipt,r);print(json.dumps(r))
if __name__=='__main__':{'register':register,'collect':collect}[sys.argv[1]]()
