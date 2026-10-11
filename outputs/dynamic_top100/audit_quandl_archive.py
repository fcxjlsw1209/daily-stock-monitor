"""Recover legacy prices from public WIKI archive, retaining raw and adjusted fields."""
from pathlib import Path
import json, zipfile
import pandas as pd
import exchange_calendars as xc
from audit_price_archives import audit

P=Path(__file__).resolve().parent
pieces=[]
inventory={}
with zipfile.ZipFile(P/'additional_sources/quandl_wiki.zip') as z:
 with z.open('WIKI_PRICES.csv') as stream:
  for chunk in pd.read_csv(stream,chunksize=250000):
   for symbol,g in chunk.groupby('ticker'):
    r=inventory.setdefault(symbol,dict(symbol=symbol,rows=0,first='9999',last='0000'))
    r['rows']+=len(g);r['first']=min(r['first'],g.date.min());r['last']=max(r['last'],g.date.max())
   part=chunk[chunk.ticker.isin(['DOW','SE'])]
   if len(part):pieces.append(part)
raw=pd.concat(pieces,ignore_index=True).rename(columns={'ticker':'symbol'})
sessions=set(xc.get_calendar('XNYS',start='1970-01-01',end='2018-12-31').sessions.strftime('%Y-%m-%d'))
results=[]
for symbol,end in [('DOW','2017-08-31'),('SE','2017-02-24')]:
 d=raw[raw.symbol.eq(symbol)].copy()
 checked=audit(d,'quandl_wiki_'+symbol)
 checked['quarantine']=checked.quarantine | ~checked.date.isin(sessions) | checked.date.gt(end)
 checked.to_csv(P/'recovered_prices'/('quandl_wiki_'+symbol+'.csv.gz'),index=False,compression='gzip')
 valid=checked[~checked.quarantine]
 expected={day for day in sessions if '2017-01-01'<=day<=end}
 # Compare raw close with independently archived raw NYSE prices, never adj_close.
 old=pd.read_csv(P/'recovered_prices/kaggle_2016_raw.csv.gz')
 old=old[old.symbol.eq(symbol)&~old.quarantine.astype(str).str.lower().eq('true')]
 both=valid.merge(old[['date','close']],on='date',suffixes=('_wiki','_archive'))
 diff=(both.close_wiki/both.close_archive-1).abs()
 results.append(dict(symbol=symbol,rows=len(d),valid_rows=len(valid),quarantined=int(checked.quarantine.sum()),first=valid.date.min(),last=valid.date.max(),target_sessions=len(expected),target_missing=sorted(expected-set(valid.date)),overlap_rows=len(both),median_relative_difference=float(diff.median()),max_relative_difference=float(diff.max()),discrepant_overlap_rows=int(diff.gt(.001).sum()),identity_and_adjustments_verified=False))
(P/'additional_sources/quandl_inventory.json').write_text(json.dumps(list(inventory.values()),indent=2))
(P/'QUANDL_RECOVERY_AUDIT.json').write_text(json.dumps(dict(source='Kaggle mirror of discontinued Quandl WIKI; source fields retained, no ledger or automatic price splice',results=results),indent=2))
print(json.dumps(results,indent=2))
