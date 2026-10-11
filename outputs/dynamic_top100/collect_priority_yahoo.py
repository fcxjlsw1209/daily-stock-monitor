"""Supplement large-cap price evidence; ticker identity is not certified here."""
from pathlib import Path
import json,hashlib,time
import yfinance as yf
from audit_price_archives import audit,summary
import pandas as pd
P=Path(__file__).resolve().parent;D=P/'additional_sources/priority_yahoo';D.mkdir(exist_ok=True)
rows=json.loads((P/'historical_largecap_priority_audit.json').read_text())['rows'];out=[]
for row in rows:
 s=row['symbol'];f=D/(s+'.csv');rec={'symbol':s,'identity_verified':False,'end_exclusive':'2026-10-06'}
 try:
  if not f.exists():
   d=yf.Ticker(s).history(start='2010-01-01',end='2026-10-06',auto_adjust=False,raise_errors=True,timeout=15)
   if d.empty:raise ValueError('empty response')
   d=d.reset_index();d['Date']=pd.to_datetime(d.Date).dt.strftime('%Y-%m-%d');d.to_csv(f,index=False)
  d=pd.read_csv(f);d['symbol']=s
  if s!='SI':audit(d,'priority_yahoo_'+s)
  rec.update(status='downloaded',rows=len(d),first=d.Date.min(),last=d.Date.max(),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
 except Exception as e:rec.update(status='unavailable',error_type=type(e).__name__)
 out.append(rec);(D/'coverage.json').write_text(json.dumps(out,indent=2));print(s,rec['status'],rec.get('rows',0),flush=True);time.sleep(.5)
pd.DataFrame(summary).to_csv(P/'recovered_prices/priority_yahoo_coverage.csv',index=False)
