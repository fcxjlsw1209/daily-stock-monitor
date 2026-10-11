"""Public chart observations only; no authentication or paid export bypass."""
from pathlib import Path
from datetime import datetime,timezone
import json,re,time,hashlib
import requests,pandas as pd
from lxml import html
from prepare import P,U
O=P/'vendor_cap';S=O/'sources';S.mkdir(parents=True,exist_ok=True)
def fetch(url,path):
 if path.exists():return path.read_bytes()
 r=requests.get(url,timeout=30)
 if r.status_code==429:
  (O/'RATE_LIMIT.json').write_text(json.dumps(dict(url=url,time=datetime.now(timezone.utc).isoformat(),retry_after=r.headers.get('Retry-After'))));raise RuntimeError('429 stop source')
 r.raise_for_status();path.write_bytes(r.content);time.sleep(.25);return r.content

def parse(content):
 match=re.search(r'\bdata\s*=\s*(\[\{"d":.*?\]);',content.decode(),re.S)
 if not match:raise ValueError('No public historical chart array')
 points=json.loads(match.group(1));x=pd.DataFrame(points)
 if not {'d','m'}.issubset(x):raise ValueError('Unexpected chart schema')
 x['date']=pd.to_datetime(x.d,unit='s',utc=True).dt.strftime('%Y-%m-%d');x['market_cap_usd']=x.m*100000
 x=x[['date','market_cap_usd']].sort_values('date').drop_duplicates('date',keep='last')
 if (x.market_cap_usd<=0).any():raise ValueError('Invalid cap')
 return x

def main():
 mappings={'EA':{'url':'https://companiesmarketcap.com/electronic-arts/marketcap/','name':'Electronic Arts EA'}}
 for page in range(1,36):
  url='https://companiesmarketcap.com/'+(f'page/{page}/' if page>1 else '')
  content=fetch(url,S/f'ranking{page}.html');tree=html.fromstring(content)
  for el in tree.xpath('//*[contains(@class,"company-code")]'):
   links=el.xpath('ancestor::a[1]')
   if links:
    ticker=el.text_content().strip();mappings[ticker]={'url':'https://companiesmarketcap.com'+links[0].get('href'),'name':' '.join(links[0].text_content().split())}
 for ticker,slug in {'ANSS':'ansys','SGEN':'seattle-genetics','XLNX':'xilinx','CERN':'cerner','SPLK':'splunk','WBA':'walgreens-boots-alliance'}.items():mappings[ticker]={'url':'https://companiesmarketcap.com/'+slug+'/marketcap/','name':ticker+' historical company'}
 (O/'DIRECTORY.json').write_text(json.dumps(mappings,indent=2)+'\n')
 m=json.loads((U/'alias_resolved_inputs/MANIFEST.json').read_text());e=pd.read_csv(U/'alias_resolved_inputs/eligible.csv.gz',index_col=0)
 aliases={r['security_id']:r['symbols']+[r['price_provider_symbol']] for r in m['alias_decisions']};records=[];missing=[];failures=[]
 for s in e:
  if not e[s].any():continue
  candidates=[s,s.replace('.','-')]+aliases.get(s,[])
  if s=='GOOGL':candidates+=['GOOG']
  if s=='FER':candidates+=['FER.MC']
  key=next((v for v in candidates if v in mappings),None)
  if key is None:missing.append(s);continue
  url=mappings[key]['url'];slug=url.rstrip('/').split('/')[-2];path=S/(slug+'.html')
  try:
   content=fetch(url,path);x=parse(content);target=O/(s+'.csv');x.to_csv(target,index=False)
   row=dict(symbol=s,vendor_ticker=key,name=mappings[key]['name'],url=url,source_file=str(path),source_sha256=hashlib.sha256(content).hexdigest(),data_file=str(target),data_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),points=len(x),first=x.date.min(),last=x.date.max(),retrieved_utc=datetime.now(timezone.utc).isoformat(),frequency='mixed monthly/weekly/daily snapshots, not complete daily',point_in_time_vintage_verified=False)
   records.append(row);print(s,len(x),x.date.min(),flush=True)
  except RuntimeError:raise
  except (ValueError,requests.RequestException) as err:failures.append(dict(symbol=s,url=url,error=str(err)))
  (O/'COLLECTION.json').write_text(json.dumps(dict(records=records,missing_mapping=missing,failures=failures),indent=2)+'\n')
 (O/'COLLECTION.json').write_text(json.dumps(dict(records=records,missing_mapping=missing,failures=failures),indent=2)+'\n')
 print('complete',len(records),'unmapped',len(missing),'failed',len(failures),flush=True)
if __name__=='__main__':main()
