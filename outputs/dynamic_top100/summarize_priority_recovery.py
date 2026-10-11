"""Summarize source coverage without certifying issuer identity or PIT readiness."""
import json,datetime,sqlite3
from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parent
# SI evidence does not identify the historical Siemens ADR. Retain raw evidence,
# quarantine derived bars and exclude it from recovered-issuer counts.
for source in ['priority_kaggle2020','priority_yahoo']:
 f=P/'recovered_prices'/f'{source}_SI.csv.gz'
 if f.exists():
  d=pd.read_csv(f);d['quarantine']=True;d['identity_quarantine_reason']='Unresolved reused SI symbol; history does not establish Siemens ADR identity';d.to_csv(f,index=False,compression='gzip')
 f=P/'recovered_prices'/f'{source}_coverage.csv'
 if f.exists():
  d=pd.read_csv(f);m=d.symbol.eq('SI');d.loc[m,'quarantined']=d.loc[m,'rows'];d.loc[m,['first_valid','last_valid']]=None;d.to_csv(f,index=False)
covs=[]
for filename in ['priority_largecap_coverage.csv','priority_kaggle2020_coverage.csv','priority_yahoo_coverage.csv']:
 f=P/'recovered_prices'/filename
 if f.exists():covs.append(pd.read_csv(f))
cov=pd.concat(covs,ignore_index=True)
sec=json.loads((P/'priority_sec/coverage.json').read_text())
report=json.loads((P/'historical_largecap_priority_audit.json').read_text())
rows=[]
for old in report['rows']:
 s=old['symbol'];c=cov[cov.symbol.eq(s)];sh=next((r for r in sec if r['symbol']==s),{})
 dates=c.last_valid.dropna()
 rows.append(dict(symbol=s,company=old['company'],source_price_rows=int(c.rows.sum()),quarantined_rows=int(c.quarantined.sum()),latest_recovered=dates.max() if len(dates) else None,share_fact_rows=sh.get('facts',0),first_share_filed=sh.get('first_filed'),identity_resolved=False,rank_ready=False))
y=json.loads((P/'additional_sources/priority_yahoo/coverage.json').read_text())
summary={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'targeted_symbols':len(rows),'price_rows_include_overlaps':True,'source_price_rows':int(cov.rows.sum()),'quarantined_rows':int(cov.quarantined.sum()),'symbols_with_price_evidence':int(cov.loc[cov.rows>cov.quarantined,'symbol'].nunique()),'yahoo_downloads_excluding_SI':sum(r['status']=='downloaded' and r['symbol']!='SI' for r in y),'sec_issuers_downloaded':sum(r['status']=='downloaded' for r in sec),'issuers_with_share_facts':sum(r.get('facts',0)>0 for r in sec),'share_fact_rows':sum(r.get('facts',0) for r in sec),'rank_ready':0,'identity_quarantine':['SI'],'rows':rows}
(P/'priority_recovery_summary.json').write_text(json.dumps(summary,indent=2));pd.DataFrame(rows).to_csv(P/'priority_recovery_gaps.csv',index=False)
with sqlite3.connect(P/'free_evidence.sqlite') as db:
 pd.DataFrame(rows).to_sql('priority_largecap_gaps',db,if_exists='replace',index=False)
 cov.to_sql('priority_price_source_coverage',db,if_exists='replace',index=False)
s=json.loads((P/'STATUS.json').read_text());s['priority_recovery']={k:v for k,v in summary.items() if k!='rows'};(P/'STATUS.json').write_text(json.dumps(s,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
