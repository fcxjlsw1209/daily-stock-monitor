"""Read-only discovery prioritization for frozen candidate dates; never events."""
import collections,datetime as dt,json
import core as c
from package import OWNED

def main():
 m=c.ROOT/'outputs/monthly_architecture_restart/META.json';f=json.loads(m.read_text())['monthly'];q=c.HERE.parent/'SEC_REVIEW_QUEUE.json';hints=json.loads(q.read_text())['candidates'];events=json.loads((c.HERE/'events.json').read_text());rows=[]
 for asof,syms in zip(f['dates'],f['symbols']):
  for s in set(syms)&set(OWNED):
   rr=sorted([r for r in events if r['ticker']==s and r['release_date']<asof],key=lambda x:x['release_date']);last=rr[-1]if rr else None
   hh=sorted([h for h in hints if h['symbol']==s and h['filing_date']<=asof],key=lambda x:x['filing_date']);h=hh[-1]if hh else None
   matched=bool(h and any(abs((dt.date.fromisoformat(r['release_date'])-dt.date.fromisoformat(h['report_date_raw']or h['filing_date']))).days<=7 for r in rr))
   reason=None
   if not last:reason='No admitted release available before candidate date'
   elif h and not matched and h['filing_date']>last['release_date']:reason='Later cached Item 2.02 candidate needs original-release review; may be guidance, preliminary or amendment, not automatically a missing quarter'
   elif (dt.date.fromisoformat(asof)-dt.date.fromisoformat(last['release_date'])).days>110:reason='Latest admitted release older than 110 days; investigate intervening scheduled quarter, not accepted as complete'
   if reason:rows.append(dict(ticker=s,candidate_date=asof,latest_admitted_release=last['release_date']if last else None,latest_admitted_fyq=f"{last['fiscal_year']}Q{last['fiscal_quarter']}"if last else None,reason=reason,discovery_hint=h,eligible_event=False,filing_date_is_not_publication=True))
 priority=collections.Counter(r['ticker']for r in rows);c.dump(c.HERE/'CANDIDATE_GAPS.json',dict(frozen_candidate_input=dict(path=c.rel(m),sha256=c.sha(m.read_bytes())),offline_discovery_hint_input=dict(path=c.rel(q),sha256=c.sha(q.read_bytes())),events_sha256=c.sha((c.HERE/'events.json').read_bytes()),policy='Planning flags only, never fabricated events. Later Item 2.02 candidates must be checked against issuer original text. The 110-day flag cannot certify latest-quarter completeness; even unflagged windows may have a missing recently published quarter.',priority_counts=dict(priority.most_common()),flagged_candidate_windows=rows))
 print('CANDIDATE PRIORITIES',dict(priority.most_common()))
if __name__=='__main__':main()
