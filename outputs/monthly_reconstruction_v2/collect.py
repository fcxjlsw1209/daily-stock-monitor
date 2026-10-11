"""Small public ETF dataset extension. One attempt per uncached symbol; halt on429."""
from pathlib import Path
import sys,json,hashlib,datetime,time,urllib.request,urllib.error
import pandas as pd,numpy as np
P=Path(__file__).resolve().parent
SYMBOLS=['SMH','SOXX','IGV','VGT','MTUM','ARKK','SOXL']

def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def main():
 manifest=[];failure=None;cutoff=pd.Timestamp('2026-09-25',tz='UTC');start=pd.Timestamp('2008-01-01',tz='UTC')
 for symbol in SYMBOLS:
  target=P/'sources'/f'{symbol}.csv';raw=P/'sources'/f'{symbol}.json';url=f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={int(start.timestamp())}&period2={int(cutoff.timestamp())}&interval=1d&events=div%2Csplits'
  if not raw.exists():
   req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
   try:
    with urllib.request.urlopen(req,timeout=30) as r:body=r.read()
   except urllib.error.HTTPError as e:
    status=dict(symbol=symbol,http_status=e.code,retry_after=e.headers.get('Retry-After'),time=datetime.datetime.now(datetime.timezone.utc).isoformat());(P/'COLLECTION_STATUS.json').write_text(json.dumps(status,indent=2)+'\n');print(status,flush=True)
    if e.code==429:failure=status;break
    continue
   except Exception as e:print(symbol,type(e).__name__,str(e)[:150],flush=True);continue
   raw.write_bytes(body)
  data=json.loads(raw.read_text());result=data.get('chart',{}).get('result')
  if not result:print(symbol,'no result',flush=True);continue
  v=result[0];ts=v.get('timestamp',[]);z=v['indicators']['quote'][0];adj=v['indicators'].get('adjclose',[{}])[0].get('adjclose',[])
  if len(adj)!=len(ts):print(symbol,'missing adjusted prices',flush=True);continue
  idx=pd.to_datetime(ts,unit='s',utc=True).tz_convert('America/New_York').tz_localize(None).normalize();df=pd.DataFrame(z,index=idx).rename(columns=str.title);df['Adj Close']=adj;df.index.name='Date'
  df=df[['Open','High','Low','Close','Adj Close','Volume']];df=df.loc[df.index<cutoff.tz_localize(None)]
  valid=df[['Open','High','Low','Close','Adj Close']].apply(np.isfinite).all(axis=1)&df['Close'].gt(0)&df.High.ge(df[['Open','Close','Low']].max(axis=1)-1e-5)&df.Low.le(df[['Open','Close']].min(axis=1)+1e-5)
  assert df.index.is_unique;df=df.loc[valid];df.to_csv(target)
  manifest.append(dict(symbol=symbol,file=str(target),url=url,raw_sha256=sha(raw),sha256=sha(target),rows=len(df),first=str(df.index.min().date()),last=str(df.index.max().date()),excluded_rows=int((~valid).sum()),source='Yahoo chart API; already known later vintage, not certified historical availability'))
  (P/'ETF_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(symbol,len(df),str(df.index[-1].date()),flush=True)
 (P/'COLLECTION_STATUS.json').write_text(json.dumps(dict(completed=[r['symbol'] for r in manifest],requested=SYMBOLS,stopped_due_to=failure),indent=2)+'\n')
if __name__=='__main__':main()
