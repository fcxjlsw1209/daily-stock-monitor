from pathlib import Path
import yfinance as yf
import pandas as pd
import hashlib,json
from curl_cffi import requests
P=Path(__file__).resolve().parent;D=P/'data';D.mkdir(exist_ok=True)
meta={}
for s in ['QQQ','QLD','TQQQ','^IRX']:
 h=yf.Ticker(s).history(start='1999-01-01',end='2026-09-26',auto_adjust=False,actions=True,raise_errors=True,timeout=30)
 if h.empty:raise ValueError(s)
 h.index=pd.to_datetime(h.index).tz_convert('America/New_York').tz_localize(None).normalize()
 p=D/(s.replace('^','')+'.csv');h.to_csv(p)
 meta[s]={'start':str(h.index.min().date()),'end':str(h.index.max().date()),'rows':len(h),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 print(s,meta[s],flush=True)
(D/'manifest.json').write_text(json.dumps(meta,indent=2))
