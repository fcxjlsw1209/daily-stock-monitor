"""Independent current-universe data audit; historical members are unverified candidates."""
from pathlib import Path
import json,hashlib,sqlite3
from datetime import datetime,timezone
import pandas as pd
import yfinance as yf
P=Path(__file__).resolve().parent
cur=pd.read_csv(P/'sources/current_sp500.csv');hist=pd.read_csv(P/'sources/historical_sp500.csv');hist['date']=pd.to_datetime(hist.date)
current=set(cur.Symbol.str.replace('.','-',regex=False));past=set(','.join(hist.loc[hist.date>='2010-01-01','tickers']).replace('.', '-').split(','));removed=sorted(past-current)
# Probe delisted/renamed names separately; never silently exclude them from a PIT backtest.
probes=[s for s in ['ATVI','ABMD','SIVB','FRC','TWTR','CERN','CELG','ALXN','SPLK','DISH'] if s in past]
symbols=sorted(current)+probes
(P/'raw').mkdir(exist_ok=True)
rows=[]
for start in range(0,len(symbols),25):
 batch=symbols[start:start+25]
 data=yf.download(batch,start='2010-01-01',end='2026-10-06',auto_adjust=False,actions=True,group_by='ticker',threads=4,progress=False,timeout=20)
 for s in batch:
  try:
   d=data[s].dropna(subset=['Close']);d.index=pd.to_datetime(d.index).tz_localize(None);d.index.name='Date'
   if d.empty:raise ValueError('no prices')
   d.to_csv(P/'raw'/f'{s}.csv.gz',compression='gzip')
   bad=(d[['Open','High','Low','Close','Adj Close']]<=0).any(axis=1)|d[['Open','High','Low','Close','Adj Close']].isna().any(axis=1)
   rows.append({'symbol':s,'group':'current' if s in current else 'removed_probe','rows':len(d),'first':str(d.index.min().date()),'last':str(d.index.max().date()),'bad_rows':int(bad.sum()),'status':'downloaded'})
  except Exception as e:rows.append({'symbol':s,'group':'current' if s in current else 'removed_probe','status':'missing','error':str(e)})
 pd.DataFrame(rows).to_csv(P/'coverage.csv',index=False)
 print('processed',len(rows),'/',len(symbols),flush=True)
with sqlite3.connect(P/'research_prices.sqlite') as db:
 cur.to_sql('current_constituents',db,if_exists='replace',index=False)
 hist.assign(date=hist.date.dt.strftime('%Y-%m-%d')).to_sql('unverified_historical_membership',db,if_exists='replace',index=False)
 frames=[]
 for row in rows:
  if row['status']=='downloaded':
   d=pd.read_csv(P/'raw'/f'{row["symbol"]}.csv.gz');d.insert(0,'symbol',row['symbol']);frames.append(d)
 if frames:pd.concat(frames).to_sql('daily_prices',db,if_exists='replace',index=False)
 db.execute('CREATE INDEX IF NOT EXISTS price_symbol_date ON daily_prices(symbol,Date)')
manifest={'created':datetime.now(timezone.utc).isoformat(),'current_count':len(current),'historical_last':str(hist.date.max().date()),'removed_since_2010':len(removed),'removed_probes':probes,'point_in_time_ready':False,'limitations':['Historical identity and effective dates not verified','Current sectors are not historical sectors','Missing delisting returns and borrow availability','Current-universe backtests have survivorship bias'],'sources':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (P/'sources').iterdir()},'source_urls':['https://github.com/datasets/s-and-p-500-companies','https://github.com/fja05680/sp500']}
(P/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))
