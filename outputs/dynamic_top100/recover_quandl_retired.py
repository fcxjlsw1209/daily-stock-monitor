"""Extract remaining historical candidates without claiming ticker identity continuity."""
from pathlib import Path
import json,zipfile
import pandas as pd
import exchange_calendars as xc
from audit_price_archives import audit
P=Path(__file__).resolve().parent
inventory=pd.read_csv(P/'historical_candidate_inventory.csv')
targets=set(inventory.loc[~inventory.current_sp500,'symbol'])-{'DOW','SE'}
parts=[]
with zipfile.ZipFile(P/'additional_sources/quandl_wiki.zip') as z:
 with z.open('WIKI_PRICES.csv') as stream:
  for chunk in pd.read_csv(stream,chunksize=250000):
   selected=chunk[chunk.ticker.isin(targets)&chunk.date.ge('2008-01-01')]
   if len(selected):parts.append(selected)
d=pd.concat(parts,ignore_index=True).rename(columns={'ticker':'symbol'})
sessions=set(xc.get_calendar('XNYS',start='2008-01-01',end='2018-12-31').sessions.strftime('%Y-%m-%d'))
out=[]
for symbol,g in d.groupby('symbol'):
 checked=audit(g,'quandl_wiki_'+symbol)
 checked['quarantine']=checked.quarantine | ~checked.date.isin(sessions)
 checked.to_csv(P/'recovered_prices'/('quandl_wiki_'+symbol+'.csv.gz'),index=False,compression='gzip')
 good=checked[~checked.quarantine]
 out.append(dict(symbol=symbol,rows=len(checked),quarantined=int(checked.quarantine.sum()),first=good.date.min(),last=good.date.max(),identity_and_adjustments_verified=False))
(P/'QUANDL_RETIRED_RECOVERY.json').write_text(json.dumps(dict(requested=len(targets),recovered=len(out),missing=sorted(targets-set(d.symbol)),coverage=out,scope='Presence evidence only; security episodes and terminal events remain separate validation'),indent=2))
print('retired candidates requested',len(targets),'recovered',len(out),'rows',len(d))
