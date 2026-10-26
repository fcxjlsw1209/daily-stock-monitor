"""Separate public-source collection only; no strategy imports or calculations."""
import argparse
from datetime import datetime, timezone
import hashlib, json, re
from pathlib import Path
import urllib.request, urllib.error, urllib.parse
from zoneinfo import ZoneInfo

ROOT = Path('/Users/shiweili/code/monthly-stock-research')
OUT = ROOT/'outputs/monthly_etf_primary_flow_sources/ssga'

def main():
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('url');args=p.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]+',args.label)
    u=urllib.parse.urlparse(args.url)
    assert u.scheme=='https' and u.hostname in ('www.ssga.com','www.spdrgoldshares.com','api.spdrgoldshares.com') and not u.username
    day=datetime.now(ZoneInfo('America/Los_Angeles')).date().isoformat()
    OUT.mkdir(parents=True,exist_ok=True)
    stop=OUT/('STOP_'+u.hostname+'_'+day+'.json')
    if stop.exists(): raise RuntimeError('Provider stopped today; reuse saved response and do not retry')
    receipt=OUT/(args.label+'_RETRIEVAL.json')
    if receipt.exists():
        data=json.loads(receipt.read_text());raw=OUT/data['raw_file']
        assert raw.exists() and hashlib.sha256(raw.read_bytes()).hexdigest()==data['sha256']
        print(json.dumps({'reused':True,'label':args.label,'status':data['status'],'bytes':data['bytes']}));return
    record={'requested_url':args.url,'requested_utc':datetime.now(timezone.utc).isoformat(),
            'local_date':day,'source_role':'public issuer raw data only; not factor admission'}
    request=urllib.request.Request(args.url,headers={'User-Agent':'LocalResearch/1.0','Accept':'*/*'})
    try:
        response=urllib.request.urlopen(request,timeout=40)
    except urllib.error.HTTPError as exc:
        response=exc
    except Exception as exc:
        error=OUT/(args.label+'_TRANSPORT_ERROR.json')
        record.update(error_type=type(exc).__name__,error=str(exc))
        with error.open('x') as f: json.dump(record,f,indent=2)
        raise
    with response:
        body=response.read(30_000_001)
        if len(body)>30_000_000:raise ValueError('Unexpected source size')
        record.update(status=response.code,final_url=response.url,
            headers={k:response.headers.get(k) for k in ('Content-Type','Last-Modified','ETag','Date','Retry-After')},
            retrieved_utc=datetime.now(timezone.utc).isoformat(),bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
    ext='.xlsx' if body[:2]==b'PK' else '.raw'
    raw=OUT/(args.label+ext)
    with raw.open('xb') as f:f.write(body)
    record['raw_file']=raw.name
    with receipt.open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
    if record['status'] in (403,429):
        with stop.open('x') as f:json.dump({'status':record['status'],'receipt':receipt.name,'policy':'Stop this host for the rest of the local day; no bypass or retry.'},f,indent=2)
    print(json.dumps({'label':args.label,'status':record['status'],'bytes':len(body),'raw_file':raw.name,'sha256':record['sha256']}))

if __name__=='__main__':main()
