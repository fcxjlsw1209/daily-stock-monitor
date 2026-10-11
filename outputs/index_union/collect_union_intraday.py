"""Recover union-specific candidates from already inventoried public archive.

Prices remain adjusted intraday proxies; bankruptcy boundaries are not bridged.
"""
from pathlib import Path
import datetime,hashlib,json,sys
import pandas as pd
import requests
import exchange_calendars as xc
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'dynamic_top100'))
from collect_azure_prices import regular_daily, BASE
A=P.parent/'dynamic_top100/additional_sources/azure_firstrate'
def main():
 inventory=json.loads((A/'inventory.json').read_text());folder=P/'sources/union_intraday';folder.mkdir(exist_ok=True)
 symbols=['ALXN','LMCK','MXIM','NIHD','CA','CTXS','ESRX','FOX','MON','YHOO','TWX','QVCA','KRFT','SNDK','WFM','KFT','LMCA'];results=[]
 schedule=xc.get_calendar('XNYS',start='2005-01-01',end='2023-01-01').schedule
 for symbol in symbols:
  matches=[r for r in inventory if r['name'] in ['FirstRate30min/'+symbol+'-DELISTED_30min.txt','FirstRate30min/'+symbol+'_30min.txt']]
  if len(matches)!=1:raise ValueError('Ambiguous archive symbol '+symbol)
  row=matches[0]
  f=folder/Path(row['name']).name;prov=f.with_suffix('.provenance.json')
  if not f.exists():
   response=requests.get(BASE+row['name'],timeout=90);response.raise_for_status()
   if len(response.content)!=row['bytes']:raise ValueError('Size changed')
   f.write_bytes(response.content);prov.write_text(json.dumps(dict(source=BASE+row['name'],sha256=hashlib.sha256(response.content).hexdigest(),retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
  meta=json.loads(prov.read_text())
  if hashlib.sha256(f.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Checksum mismatch')
  raw=pd.read_csv(f,header=None,names=['timestamp','open','high','low','close','volume']);d=regular_daily(raw,schedule)
  d['adj_close']=d.close;d['symbol']=symbol;d['identity_verified']=False
  dest=folder/(symbol+'.csv');d.to_csv(dest,index=False)
  dest.with_suffix('.provenance.json').write_text(json.dumps(dict(source=meta,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),identity_verified=False,adjustment_verified=False),indent=2))
  results.append(dict(symbol=symbol,rows=len(d),first=d.date.min(),last=d.date.max(),pre2012=int(d.date.lt('2012-01-01').sum())))
  print(results[-1],flush=True)
 (folder/'INVENTORY.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
