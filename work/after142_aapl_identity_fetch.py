"""Separate public-source acquisition, never imported into an offline replay."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import urllib.request,urllib.error
R=Path(__file__).resolve().parents[1]
D=R/'outputs/monthly_aapl_identity_refetch_v1'
urls=['https://www.nasdaqtrader.com/TraderNews.aspx?id=dtn2014-16',
      'https://www.nasdaqtrader.com/TraderNews.aspx?id=dtn2019-04']
D.mkdir(exist_ok=False);(D/'raw').mkdir()
plan={'created_utc':datetime.now(timezone.utc).isoformat(),'urls':urls,
 'scope':'new public retrieval only; never restore or claim original hashes',
 'stop_rule':'on any HTTP error/429, no further requests this run; honor Retry-After',
 'requests_max':2,'credentials_used':False,'offline_replay':False,'model_calls':0}
(D/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
rows=[]
for i,url in enumerate(urls):
 t=datetime.now(timezone.utc).isoformat()
 req=urllib.request.Request(url,headers={'User-Agent':'monthly-stock-research/1.0 (public historical data research)','Accept-Encoding':'identity'})
 try:
  with urllib.request.urlopen(req,timeout=45) as response:
   raw=response.read(2*1024*1024+1)
   if len(raw)>2*1024*1024:raise ValueError('oversized response')
   status=response.status;final=response.geturl()
   hdr={k:response.headers.get(k) for k in ['Content-Type','Content-Encoding','Date','Last-Modified','Retry-After'] if response.headers.get(k)}
 except urllib.error.HTTPError as e:
  raw=e.read(2*1024*1024);status=e.code;final=e.geturl()
  hdr={k:e.headers.get(k) for k in ['Content-Type','Retry-After','Date'] if e.headers.get(k)}
 except Exception as e:
  rows.append({'url':url,'attempted_utc':t,'error':type(e).__name__+': '+str(e),'stopped':True});break
 name='source'+str(i+1);p=D/'raw'/(name+'.response');p.write_bytes(raw)
 row={'url':url,'final_url':final,'attempted_utc':t,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
  'http_status':status,'headers':hdr,'response_path':str(p.relative_to(R)),
  'response_sha256':hashlib.sha256(raw).hexdigest(),'response_bytes':len(raw),'public_unauthenticated':True,
  'original_old_bytes_recovered':False}
 (D/'raw'/(name+'.meta.json')).write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
 if status!=200 or final!=url:
  row['stopped']=True;break
summary={'created_utc':datetime.now(timezone.utc).isoformat(),'rows':rows,'requested':len(rows),'strategy_trials':0,
 'all_http200':len(rows)==2 and all(r.get('http_status')==200 and r.get('final_url')==r['url'] for r in rows),
 'identity_admitted':False,'first_public_version_verified':False}
(D/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
