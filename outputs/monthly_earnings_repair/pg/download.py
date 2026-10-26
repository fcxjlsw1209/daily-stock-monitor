"""Download issuer-linked full filing HTMLs; filing metadata is discovery only."""
import json
from datetime import datetime
import collect as c
rows=json.loads((c.RAW/'issuer_8k_feed_cik.json').read_text())['GetEdgarFilingListResult']
offline=json.loads((c.HERE.parent/'SEC_REVIEW_QUEUE.json').read_text())['candidates']
dates={r['filing_date'] for r in offline if r['symbol']=='PG'}
selected=[]
for row in rows:
    date=datetime.strptime(row['FilingDate'],'%m/%d/%Y %H:%M:%S').date()
    if not '2011-01-01'<=date.isoformat()<=c.CUTOFF: continue
    if date.isoformat() in dates or (date.year<2015 and ((date.month in (1,4,7,10) and date.day>=20) or (date.month==8 and date.day<=10))): selected.append(row)
c.dump(c.HERE/'DISCOVERY.json',dict(source='issuer_8k_feed_cik.json',discovery_only=True,metadata_is_not_release_time=True,selection='2015+ cached item 2.02 candidate filing dates; 2011-2014 seasonal date heuristic only to locate bodies, not admit events',selected=selected))
print('candidates',len(selected),flush=True)
for row in selected:
    url=next(x['Url'] for x in row['DocumentList'] if x['DocumentType']=='HTML')
    try:
        b,p,m=c.fetch(url,'filing_'+str(row['FilingId']))
        print(row['FilingDate'],row['FilingId'],len(b),flush=True)
    except Exception as e:
        print('ERROR',row['FilingId'],str(e),flush=True)
        if (c.HERE/'SOURCE_STOPS.json').exists(): break
