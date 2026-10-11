"""Cache selected files from public 2020 Yahoo snapshot, retaining provenance."""
from pathlib import Path
from urllib.parse import quote
import hashlib,io,json,zipfile,datetime
import requests
import pandas as pd
P=Path(__file__).resolve().parent
D=P/'sources/kaggle2020';D.mkdir(exist_ok=True)
def main():
 results=[]
 for symbol in ['QRTEA','FOXA','FOX','CEPH','NIHD','MICC']:
  f=D/(symbol+'.csv');url='https://www.kaggle.com/api/v1/datasets/download/jacksoncrow/stock-market-dataset/'+quote('stocks/'+symbol+'.csv',safe='')
  if not f.exists():
   r=requests.get(url,timeout=45)
   if r.status_code in [429,503]:print('rate_limited_stop',flush=True);break
   if r.status_code!=200:results.append(dict(symbol=symbol,status=r.status_code));continue
   payload=r.content
   if zipfile.is_zipfile(io.BytesIO(payload)):
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
     names=[x for x in z.infolist() if Path(x.filename).name==symbol+'.csv']
     if len(names)!=1 or names[0].file_size>64000000:raise ValueError('Unexpected archive')
     payload=z.read(names[0])
   if not payload.startswith(b'Date,'):results.append(dict(symbol=symbol,status='not_csv'));continue
   f.write_bytes(payload)
   f.with_suffix('.provenance.json').write_text(json.dumps(dict(source=url,sha256=hashlib.sha256(payload).hexdigest(),retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),identity_verified=False),indent=2))
  d=pd.read_csv(f);results.append(dict(symbol=symbol,rows=len(d),first=d.Date.min(),last=d.Date.max()));print(results[-1],flush=True)
 (D/'INVENTORY.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
