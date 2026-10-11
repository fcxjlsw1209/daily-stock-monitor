"""Collect public facts for a historical candidate sample, not a certified universe."""
from pathlib import Path
import pandas as pd,requests,json,zipfile,time,concurrent.futures,hashlib
P=Path(__file__).resolve().parent;A=P/'additional_sources';D=P/'expanded_sec';D.mkdir(exist_ok=True)
rows=[]
def add(s,c,source,name):
 try:c=str(int(float(c))).zfill(10)
 except (ValueError,TypeError):return
 if c=='0000000000':return
 rows.append(dict(symbol=s,cik=c,source=source,name=name,identity_verified=False))
cur=pd.read_csv(P.parent/'universe_expansion/sources/current_sp500.csv')
for _,r in cur.iterrows():add(r.Symbol,r.CIK,'current_sp500',r.Security)
with zipfile.ZipFile(A/'kaggle_nyse.zip') as z:
 old=pd.read_csv(z.open('securities.csv'))
for _,r in old.iterrows():add(r['Ticker symbol'],r.CIK,'kaggle_2016_roster',r.Security)
hist=set(pd.read_csv(P/'historical_candidate_inventory.csv').symbol)
delists=pd.read_csv(A/'delist_delistings.txt')
for _,r in delists.iterrows():
 if str(r.ticker).replace('.','-') in hist:add(r.ticker,r.cik,'delist_sample_symbol_match_unverified',r.resolved_name)
roster=pd.DataFrame(rows).drop_duplicates(['symbol','cik','source']);roster.to_csv(D/'candidate_identity_evidence.csv',index=False)
ciks=sorted(set(roster.cik));print('candidate CIKs',len(ciks),flush=True)
def fetch(cik):
 path=D/(cik+'.json');prior=P/'sec_coverage'/(cik+'.json');rec=dict(cik=cik)
 try:
  if prior.exists():path=prior
  elif not path.exists():
   r=requests.get('https://data.sec.gov/api/xbrl/companyfacts/CIK'+cik+'.json',headers={'User-Agent':'Personal investment research data-quality audit'},timeout=30)
   r.raise_for_status();data=r.json();assert 'facts' in data;path.write_bytes(r.content);time.sleep(.5)
  data=json.loads(path.read_text());facts=[]
  for tax,concepts in data['facts'].items():
   for name,item in concepts.items():
    if name in ['EntityCommonStockSharesOutstanding','CommonStockSharesOutstanding','NumberOfSharesOutstanding']:
     facts.extend(dict(v,taxonomy=tax,concept=name,cik=cik) for v in item.get('units',{}).get('shares',[]))
  if facts:pd.DataFrame(facts).to_csv(D/(cik+'_shares.csv'),index=False)
  rec.update(status='downloaded',facts=len(facts),path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),entityName=data.get('entityName'))
 except Exception as e:rec.update(status='error',error=str(e))
 return rec
out=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
 for rec in ex.map(fetch,ciks):
  out.append(rec);(D/'coverage.json').write_text(json.dumps(out,indent=2))
  if len(out)%25==0:print(len(out),'/',len(ciks),'downloaded',sum(x['status']=='downloaded' for x in out),flush=True)
print('Finished',len(out),'issuers',sum(x.get('facts',0) for x in out),'facts',flush=True)
