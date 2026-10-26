"""Read-only evaluation of parent-prioritized candidate history; writes batch audit only."""
from finalize import *
def audit():
 source=HERE.parent/'CANDIDATE_HISTORY_AUDIT.json';j=json.load(open(source));es=json.load(open(HERE/'events.json'))
 affected={'IBM','HPQ','QCOM','CSCO','INTC'};out=[]
 for r in j['review_rows']:
  if r['scope']!='training' or r['symbol'] not in affected:continue
  options=[]
  for e in es:
   if e['ticker']!=r['symbol']:continue
   n=validate_event(e)
   if n['available_session']<=r['signal_date']:options.append(e)
  last=max(options,key=lambda e:e['release_date']) if options else None
  age=(dt.date.fromisoformat(r['signal_date'])-dt.date.fromisoformat(last['release_date'])).days if last else None
  out.append(dict(r,updated_prior_release=last['release_date'] if last else None,updated_age_days=age,updated_state='no_prior_verified_event' if last is None else 'review_reporting_interval_over_120d' if age>120 else 'prior_event_within_120d'))
 counts={k:dict(rows=sum(r['symbol']==k for r in out),remaining=sum(r['symbol']==k and r['updated_state']!='prior_event_within_120d' for r in out)) for k in sorted(affected)}
 dump(HERE/'TRAINING_PRIORITY_AUDIT.json',dict(status='evidence_priority_only_not_backtest_gate',input_path=rel(source),input_sha256=sha(source.read_bytes()),events_sha256=sha((HERE/'events.json').read_bytes()),generated_at=now(),per_ticker=counts,rows=out,limitation='Original parent-prioritized review rows only; conservative date-only availability uses central schema next-session close. 120 days is an audit heuristic, not complete-release-history certification.'))
 print(counts)
if __name__=='__main__':audit()
