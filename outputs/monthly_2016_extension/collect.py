from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,yfinance as yf
P=Path(__file__).resolve().parent/'prices';P.mkdir(exist_ok=True)
today=datetime.now(timezone.utc).date().isoformat()
limit=P/'RATE_LIMIT.json'
if limit.exists() and json.loads(limit.read_text())['day']==today:raise RuntimeError('Rate limited today')
for s in ['BUD','NVS','HSBC']:
 f=P/(s+'.csv')
 if f.exists():print('REUSE',s,flush=True);continue
 try:
  z=yf.Ticker(s).history(start='2008-01-01',end='2026-09-25',auto_adjust=False,actions=True,raise_errors=True)
  assert not z.empty and 'Adj Close' in z
  z.index=z.index.tz_localize(None).normalize();assert z.index.is_unique
  z.to_csv(f)
  f.with_suffix('.source.json').write_text(json.dumps(dict(symbol=s,source='Yahoo/yfinance',retrieved_utc=datetime.now(timezone.utc).isoformat(),start='2008-01-01',end_exclusive='2026-09-25',sha256=hashlib.sha256(f.read_bytes()).hexdigest(),rows=len(z)),indent=2)+'\n')
  print('SAVED',s,len(z),flush=True)
 except Exception as ex:
  if '429' in str(ex) or 'RateLimit' in type(ex).__name__:
   limit.write_text(json.dumps(dict(day=today,symbol=s))+'\n');raise
  print('FAIL',s,type(ex).__name__,str(ex)[:150],flush=True)
