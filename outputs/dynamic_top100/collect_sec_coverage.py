"""Coverage probe of known issuers, not a complete historical universe."""
from pathlib import Path
import json,time,sqlite3
import pandas as pd
import requests
P=Path(__file__).resolve().parent;D=P/'sec_coverage';D.mkdir(exist_ok=True)
m=pd.read_csv(P.parent/'universe_expansion/sources/current_sp500.csv');m['Symbol']=m.Symbol.str.replace('.','-',regex=False)
with sqlite3.connect(f'file:{P.parent}/stock_database/stocks.sqlite?mode=ro',uri=True) as db:old=set(pd.read_sql_query('select symbol from companies',db).symbol)
m=m[m.Symbol.isin(old)];out=[]
for _,row in m.iterrows():
 s=row.Symbol;cik=str(int(row.CIK)).zfill(10);path=D/(cik+'.json');record={'symbol':s,'cik':cik}
 try:
  if not path.exists():
   r=requests.get('https://data.sec.gov/api/xbrl/companyfacts/CIK'+cik+'.json',headers={'User-Agent':'Personal investment research data-quality audit'},timeout=20)
   r.raise_for_status();path.write_bytes(r.content);time.sleep(.25)
  data=json.loads(path.read_text());facts=[]
  for tax,concepts in data['facts'].items():
   for name,item in concepts.items():
    if name in ['EntityCommonStockSharesOutstanding','CommonStockSharesOutstanding','NumberOfSharesOutstanding']:
     facts.extend(dict(v,taxonomy=tax,concept=name) for v in item.get('units',{}).get('shares',[]))
  d=pd.DataFrame(facts);d.to_csv(D/(s+'_shares.csv'),index=False)
  record.update(facts=len(d),first_end=d.end.min() if len(d) else None,first_filed=d.filed.min() if len(d) else None,last_filed=d.filed.max() if len(d) else None,status='downloaded')
 except Exception as e:record.update(status='error',error=str(e).split('?')[0])
 out.append(record);pd.DataFrame(out).to_csv(D/'coverage.csv',index=False)
 if len(out)%10==0:print('processed',len(out),'/',len(m),flush=True)
print('Complete',len(out),'issuers;',sum(x['status']=='downloaded' for x in out),'downloaded',flush=True)
