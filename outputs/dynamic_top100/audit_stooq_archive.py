from pathlib import Path
import pandas as pd,zipfile,json,hashlib
from audit_price_archives import audit,summary,D
P=Path(__file__).resolve().parent;A=P/'additional_sources'
hist=set(pd.read_csv(P/'historical_candidate_inventory.csv').symbol.str.upper())
with zipfile.ZipFile(A/'kaggle_stooq_2017.zip') as z:
 names=[n for n in z.namelist() if n.startswith('Stocks/') and n.endswith('.txt')]
 inventory=[]
 for name in names:
  symbol=Path(name).name.replace('.us.txt','').upper();info=z.getinfo(name)
  row=dict(symbol=symbol,path=name,bytes=info.file_size,selected=symbol.replace('.','-') in hist,identity_verified=False)
  if row['selected'] and info.file_size:
   raw=z.read(name);row['sha256']=hashlib.sha256(raw).hexdigest()
   try:
    df=pd.read_csv(z.open(name));df['symbol']=symbol;audit(df,'stooq2017_'+symbol.replace('/','_'));row['rows']=len(df)
   except Exception as e:row['error']=str(e)
  inventory.append(row)
  if len(inventory)%500==0:print('audited inventory',len(inventory),'/',len(names),flush=True)
 pd.DataFrame(inventory).to_csv(A/'stooq_archive_inventory.csv',index=False)
pd.DataFrame(summary).to_csv(D/'stooq_coverage.csv',index=False)
print('stocks',len(inventory),'selected',sum(r['selected'] for r in inventory),'rows',sum(r.get('rows',0) for r in inventory),'quarantined',sum(r['quarantined'] for r in summary),flush=True)
