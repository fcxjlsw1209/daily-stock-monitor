"""Read-only frozen-candidate audit; writes only the task directory."""
from common import *
import sys,pandas as pd,exchange_calendars as xc
sys.path.insert(0,str(HERE.parent))
from event_schema import normalize_event
def audit():
 cal=xc.get_calendar('XNYS',start='2010-01-01',end='2027-12-31')
 rows=json.loads((HERE.parent/'MONTHLY_AVAILABILITY.json').read_text());result=[]
 for k in CONFIG:
  ps=HERE/k/'events.json';events=json.loads(ps.read_text()) if ps.exists() else []
  normalized=[]
  for e in events:
   n=normalize_event(dict(symbol=k,official=dict(verified=True,publication_kind='release',publication_date=e['release_date'],publication_timezone=e['publication_timezone'],actual_period_end=e['fiscal_period_end_actual'],source_url=e['source_url'])),calendar=cal)
   assert n['status']=='ready',n
   normalized.append((pd.Timestamp(n['known_by_at']),e))
  missing=[];stale=[]
  for row in rows:
   if row['symbol']!=k:continue
   asof=cal.session_close(pd.Timestamp(row['signal_date']))
   eligible=[e for when,e in normalized if when<=asof]
   if not eligible:missing.append(row['signal_date']);continue
   latest=max(eligible,key=lambda e:e['release_date'])
   if (pd.Timestamp(row['signal_date'])-pd.Timestamp(latest['release_date'])).days>120:stale.append(dict(signal_date=row['signal_date'],latest_release=latest['release_date']))
  result.append(dict(ticker=k,event_count=len(events),no_prior_verified_event=missing,latest_event_older_than_120_days=stale))
 dump(HERE/'FROZEN_WINDOW_AUDIT.json',result)
 for r in result:print(r['ticker'],r['event_count'],'missing',len(r['no_prior_verified_event']),'stale',len(r['latest_event_older_than_120_days']),r['no_prior_verified_event'][:3],r['latest_event_older_than_120_days'][:3],flush=True)
if __name__=='__main__':audit()
