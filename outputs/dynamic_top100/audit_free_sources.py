"""Audit free shares evidence; never converts observations into verified PIT ranks."""
from pathlib import Path
import json,time,hashlib
import pandas as pd
import requests
P=Path(__file__).resolve().parent/'source_probes'
CIKS={'AAPL':'0000320193','MSFT':'0000789019','TSM':'0001046179','ATVI':'0000718877'}
rows=[]
for symbol,cik in CIKS.items():
 d=pd.read_csv(P/(symbol+'_yahoo_shares.csv'));d.columns=['observed_timestamp','shares'];d['date']=d.observed_timestamp.str[:10]
 grouped=d.groupby('date').shares.agg(['count','nunique','min','max']);conflicts=grouped[grouped['nunique']>1]
 conflicts.to_csv(P/(symbol+'_share_conflicts.csv'))
 row={'symbol':symbol,'yahoo_first':d.date.min(),'yahoo_last':d.date.max(),'conflict_dates':len(conflicts),'max_same_date_ratio':float((conflicts['max']/conflicts['min']).max()),'verified_pit':False}
 path=P/(symbol+'_sec_companyfacts.json')
 try:
  if not path.exists():
   response=requests.get('https://data.sec.gov/api/xbrl/companyfacts/CIK'+cik+'.json',headers={'User-Agent':'Personal investment research data-quality audit','Accept-Encoding':'gzip, deflate'},timeout=25)
   row['sec_http_status']=response.status_code
   if response.status_code!=200:raise RuntimeError('SEC HTTP '+str(response.status_code))
   path.write_bytes(response.content)
  facts=json.loads(path.read_text());entries=[]
  for taxonomy,items in facts['facts'].items():
   for concept,item in items.items():
    if concept not in ['EntityCommonStockSharesOutstanding','CommonStockSharesOutstanding','NumberOfSharesOutstanding']:continue
    for unit,values in item.get('units',{}).items():
     if unit=='shares':entries.extend(dict(v,taxonomy=taxonomy,concept=concept) for v in values)
  frame=pd.DataFrame(entries);frame.to_csv(P/(symbol+'_sec_shares.csv'),index=False)
  row.update(sec_share_facts=len(frame),sec_first_end=frame.end.min() if len(frame) else None,sec_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
  # Matching same observation date/value supports units only, not Yahoo publication timing.
  if len(frame):
   match=d.merge(frame,left_on=['date','shares'],right_on=['end','val'])
   match.to_csv(P/(symbol+'_sec_matches.csv'),index=False)
   row['exact_date_value_matches']=len(match)
 except Exception as e:row['sec_error']=str(e)
 rows.append(row);print(json.dumps(row),flush=True);time.sleep(.25)
(P/'free_source_audit.json').write_text(json.dumps(rows,indent=2))
