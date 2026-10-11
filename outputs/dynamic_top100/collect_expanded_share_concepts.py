"""Resume share-only collection through SEC official companyconcept endpoints."""
from pathlib import Path
import requests,json,time,concurrent.futures,pandas as pd,hashlib
P=Path(__file__).resolve().parent;D=P/'expanded_sec';C=D/'concepts';C.mkdir(exist_ok=True)
ciks=sorted(set(pd.read_csv(D/'candidate_identity_evidence.csv',dtype={'cik':str}).cik))
prior={x['cik']:x for x in json.loads((D/'coverage.json').read_text()) if x['status']=='downloaded'}
def fetch(cik):
 if cik in prior:return dict(prior[cik],mode='companyfacts_cached')
 facts=[];sources=[];errors=[]
 for tax,tag in [('dei','EntityCommonStockSharesOutstanding'),('us-gaap','CommonStockSharesOutstanding'),('ifrs-full','NumberOfSharesOutstanding')]:
  path=C/(cik+'_'+tax+'_'+tag+'.json');url=f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{tax}/{tag}.json'
  try:
   if not path.exists():
    r=requests.get(url,headers={'User-Agent':'Personal investment research data-quality audit'},timeout=20);time.sleep(.6)
    if r.status_code==404:errors.append(dict(url=url,status=404));continue
    r.raise_for_status();data=r.json();assert 'units' in data;path.write_bytes(r.content)
   data=json.loads(path.read_text());facts.extend(dict(v,taxonomy=tax,concept=tag,cik=cik) for v in data.get('units',{}).get('shares',[]));sources.append(dict(url=url,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
   if facts:break
  except Exception as e:errors.append(dict(url=url,error=str(e)))
 if facts:pd.DataFrame(facts).to_csv(D/(cik+'_shares.csv'),index=False)
 return dict(cik=cik,status='downloaded' if facts else 'unresolved',facts=len(facts),mode='companyconcept',sources=sources,errors=errors)
out=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 for rec in ex.map(fetch,ciks):
  out.append(rec);tmp=D/'share_coverage.tmp';tmp.write_text(json.dumps(out,indent=2));tmp.replace(D/'share_coverage.json')
  if len(out)%50==0:print(len(out),'/',len(ciks),'with facts',sum(x.get('facts',0)>0 for x in out),flush=True)
print('Finished',len(out),'facts',sum(x.get('facts',0) for x in out),flush=True)
