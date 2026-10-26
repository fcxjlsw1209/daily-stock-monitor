"""Independent read-only source-scope audit. Writes only this audit directory."""
from pathlib import Path
import json,hashlib,datetime as dt,sys
from collections import defaultdict,Counter
import pandas as pd
HERE=Path(__file__).resolve().parent
P=HERE.parent.parent
ROOT=P.parent.parent
sys.path.insert(0,str(P))
from event_schema import normalize_event
from research_calendar import market_calendar
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def qid(e):return e['fiscal_year']*4+e['fiscal_quarter']-1
def short(e):return {k:e.get(k) for k in ['symbol','publication_date','available_session','official_actual_period_end','fiscal_year','fiscal_quarter','source_url','evidence_path','evidence_sha256']}
def main():
 sources=[P/'EVENTS.json',ROOT/'outputs/monthly_architecture_restart/META.json',P/'warmup/pfe_supplement/events.json']
 blobs=[x.read_bytes() for x in sources];events,meta,pfe=[json.loads(x) for x in blobs]
 inputs={str(p):sha(b) for p,b in zip(sources,blobs)}
 dump(HERE/'INPUTS.json',dict(generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),inputs=inputs,central_count=len(events),mode='read-only audit; local PFE overlay only if not integrated'))
 (HERE/'EVENTS_INPUT.json').write_bytes(blobs[0]);(HERE/'META_INPUT.json').write_bytes(blobs[1])
 keys={(e['symbol'],e['publication_date']) for e in events}
 for e in pfe:
  if ('PFE',e['release_date']) in keys:continue
  n=normalize_event(dict(symbol='PFE',official=dict(verified=True,publication_kind='release',publication_date=e['release_date'],publication_timezone=e['publication_timezone'],actual_period_end=e['fiscal_period_end_actual'],source_url=e['source_url'])))
  n.update(fiscal_year=e['fiscal_year'],fiscal_quarter=e['fiscal_quarter'],evidence_path=e['evidence_path'],evidence_sha256=e['evidence_sha256'],audit_overlay_only=True)
  events.append(n)
 groups=defaultdict(list)
 for e in events:
  if e['status']=='ready':groups[e['symbol']].append(e)
 for a in groups.values():a.sort(key=lambda e:e['known_by_at'])
 etfs=set(meta['monthly']['etfs']);cal=market_calendar(start='2010-01-01',end='2030-12-31')
 allrows=[];issuers=defaultdict(lambda:defaultdict(list));futurebad=[];same_day_excluded=[];endpoint=[];gaps=[];missing=[]
 for scope in ['monthly','training']:
  for day,symbols in zip(meta[scope]['dates'],meta[scope]['symbols']):
   asof=cal.session_close(pd.Timestamp(day))
   for s in symbols:
    if s in etfs:continue
    a=groups[s];known=[e for e in a if pd.Timestamp(e['known_by_at'])<=asof]
    e=max(known,key=lambda x:x['known_by_at']) if known else None
    row=dict(scope=scope,symbol=s,signal_date=day,latest=short(e) if e else None)
    issuers[s][scope].append(row);allrows.append(row)
    if not e:missing.append(row);continue
    if e['publication_date']>day or e['available_session']>day or (e['precision']=='date_only' and e['publication_date']==day):futurebad.append(row)
    for same in a:
     if same['publication_date']==day and same['precision']=='date_only':
      assert same not in known
      same_day_excluded.append(dict(scope=scope,symbol=s,signal_date=day,excluded_release=short(same),selected=short(e)))
    later=[x for x in a if qid(x)>qid(e)]
    if not later:
     endpoint.append(dict(row,days_after_period=(pd.Timestamp(day)-pd.Timestamp(e['official_actual_period_end'])).days,age_release=(pd.Timestamp(day)-pd.Timestamp(e['publication_date'])).days))
    else:
     nxt=min(later,key=qid)
     if qid(nxt)-qid(e)>1:
      gaps.append(dict(row,next=short(nxt),missing_identities=[f'{n//4}Q{n%4+1}' for n in range(qid(e)+1,qid(nxt))],days_after_period=(pd.Timestamp(day)-pd.Timestamp(e['official_actual_period_end'])).days))
 summaries=[]
 for s,sc in sorted(issuers.items()):
  a=groups[s];last=max(a,key=qid) if a else None
  summaries.append(dict(symbol=s,events=len(a),earliest_release=a[0]['publication_date'] if a else None,last_covered=short(last) if last else None,scopes={k:dict(rows=len(v),first=v[0]['signal_date'],last=v[-1]['signal_date'],first_prior=v[0]['latest'],last_prior=v[-1]['latest']) for k,v in sc.items()},endpoint_candidate_count=sum(x['symbol']==s for x in endpoint),max_endpoint_days_after_period=max((x['days_after_period'] for x in endpoint if x['symbol']==s),default=None)))
 dump(HERE/'ISSUER_SCOPE.json',summaries);dump(HERE/'ENDPOINT_ROWS.json',endpoint);dump(HERE/'INTERNAL_SEQUENCE_REVIEW.json',gaps);dump(HERE/'NO_PRIOR_EVENT.json',missing);dump(HERE/'SAME_DAY_EXCLUDED.json',same_day_excluded);dump(HERE/'TEMPORAL_ERRORS.json',futurebad)
 print('central',len(json.loads(blobs[0])),'with overlay',len(events),'issuers',len(issuers),'candidate rows',len(allrows),'futurebad',len(futurebad),'same_day_excluded',len(same_day_excluded))
 print('missing',[(r['scope'],r['symbol'],r['signal_date']) for r in missing])
 print('sequence candidates',dict(Counter(x['symbol'] for x in gaps)))
 print('endpoint candidates >85 days after period')
 for s in summaries:
  if (s['max_endpoint_days_after_period'] or 0)>85:print(json.dumps(s))
 print('all endpoints',[(s['symbol'],s['last_covered']['fiscal_year'],s['last_covered']['fiscal_quarter'],s['max_endpoint_days_after_period']) for s in summaries])
if __name__=='__main__':main()

