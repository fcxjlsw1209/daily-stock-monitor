"""Conservative disclosure-date events; units/splits/classes still need verification."""
from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parent/'sec_coverage'
records=[];summary=[]
for f in sorted(P.glob('*_shares.csv')):
 try:d=pd.read_csv(f)
 except pd.errors.EmptyDataError:continue
 if d.empty or 'concept' not in d:continue
 d=d[d.concept.eq('EntityCommonStockSharesOutstanding')].copy()
 if d.empty:continue
 d=d[(pd.to_datetime(d.end)<=pd.to_datetime(d.filed))&(d.val>0)]
 count=0;bad=0
 for (filed,end),g in d.groupby(['filed','end']):
  values=g.val.unique();conflict=len(values)!=1;bad+=int(conflict);count+=1
  # Companyfacts has filing DATE, not exact acceptance timestamp: use end of NY day.
  available=(pd.Timestamp(filed)+pd.Timedelta(days=1)).tz_localize('America/New_York').tz_convert('UTC')
  records.append(dict(symbol=f.name.removesuffix('_shares.csv'),reported_as_of=end,filed=filed,available_no_earlier_than=available.isoformat(),shares=None if conflict else values[0],conflict=conflict,accessions=';'.join(sorted(set(g.accn))),unit_and_class_verified=False))
 summary.append(dict(symbol=f.name.removesuffix('_shares.csv'),events=count,conflicts=bad,first_filed=d.filed.min(),last_filed=d.filed.max()))
pd.DataFrame(records).to_csv(P/'disclosure_events_unverified.csv',index=False)
pd.DataFrame(summary).to_csv(P/'event_coverage.csv',index=False)
print('Disclosure events',len(records),'companies',len(summary),'ambiguous',sum(x['conflict'] for x in records))
