from pathlib import Path
import pandas as pd,requests,json,time,concurrent.futures,hashlib,zipfile,io
from urllib.parse import quote
P=Path(__file__).resolve().parent;D=P/'additional_sources/kaggle2020';D.mkdir(exist_ok=True)
h=pd.read_csv(P/'historical_candidate_inventory.csv');symbols=sorted(h.loc[~h.current_sp500,'symbol'].unique())
previous={r['symbol']:r for r in json.loads((D/'coverage.json').read_text())} if (D/'coverage.json').exists() else {}
def get(symbol):
 if previous.get(symbol,{}).get('status')==404:return previous[symbol]
 path=D/(symbol+'.csv');url='https://www.kaggle.com/api/v1/datasets/download/jacksoncrow/stock-market-dataset/'+quote('stocks/'+symbol+'.csv',safe='');rec=dict(symbol=symbol,url=url)
 try:
  if not path.exists():
   r=requests.get(url,timeout=20);time.sleep(.6)
   if r.status_code!=200:return dict(rec,status=r.status_code)
   payload=r.content
   if zipfile.is_zipfile(io.BytesIO(payload)):
    (D/(symbol+'.zip')).write_bytes(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
     entries=[x for x in z.infolist() if Path(x.filename).name==symbol+'.csv']
     if len(entries)!=1 or entries[0].file_size>64*1024*1024:return dict(rec,status='unexpected_archive')
     payload=z.read(entries[0])
   if not payload.startswith(b'Date,'):return dict(rec,status='not_price_csv',content_type=r.headers.get('Content-Type'))
   path.write_bytes(payload)
  d=pd.read_csv(path);rec.update(status='downloaded',rows=len(d),first=d.Date.min(),last=d.Date.max(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),identity_and_adjustments_verified=False)
 except Exception as e:rec.update(status='error',error=str(e))
 return rec
out=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
 for r in ex.map(get,symbols):
  out.append(r);(D/'coverage.json').write_text(json.dumps(out,indent=2))
  if len(out)%50==0:print(len(out),'/',len(symbols),'downloaded',sum(x['status']=='downloaded' for x in out),flush=True)
print('Finished',len(out),'downloaded',sum(x['status']=='downloaded' for x in out),flush=True)
