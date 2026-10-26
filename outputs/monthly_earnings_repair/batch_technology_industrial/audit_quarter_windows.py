"""Read-only quarter-sequence audit against actual frozen candidate dates."""
from finalize import *

def availability(e):
 return normalize_event(dict(symbol=e['ticker'],official=dict(verified=True,publication_kind='release',publication_date=e['release_date'],publication_timezone=e['publication_timezone'],actual_period_end=e['fiscal_period_end_actual'],source_url=e['source_url'])))['available_session']

def audit_events(events,meta):
 out=[];reviews=[]
 for ticker in sorted({e['ticker'] for e in events}):
  es=sorted([dict(e,available_session=availability(e)) for e in events if e['ticker']==ticker],key=lambda e:e['release_date'])
  windows=[]
  for left,right in zip(es,es[1:]):
   a=left['fiscal_year']*4+left['fiscal_quarter']-1;b=right['fiscal_year']*4+right['fiscal_quarter']-1
   if b-a>1:windows.append(dict(prior_release=left['release_date'],next_release=right['release_date'],prior_available=left['available_session'],next_available=right['available_session'],missing_quarters=[dict(fiscal_year=n//4,fiscal_quarter=n%4+1) for n in range(a+1,b)]))
  for scope in ['monthly','training']:
   days=[d for d,syms in zip(meta[scope]['dates'],meta[scope]['symbols']) if ticker in syms];affected=[];no_prior=[]
   for d in days:
    if not any(e['available_session']<=d for e in es):no_prior.append(d)
    for w in windows:
     if w['prior_available']<=d<w['next_available']:affected.append(dict(signal_date=d,**w))
   reviews.extend(dict(ticker=ticker,scope=scope,**r) for r in affected)
   out.append(dict(ticker=ticker,scope=scope,candidate_count=len(days),first_candidate=min(days) if days else None,last_candidate=max(days) if days else None,internal_gap_candidate_rows=len({r['signal_date'] for r in affected}),no_prior_verified_event_dates=no_prior))
 return dict(per_ticker_scope=out,internal_gap_reviews=reviews,method='Every adjacent observed FY/Q pair is checked for missing quarter identities; each gap is intersected with actual candidate signal dates using conservative central-schema availability, independently of the 120-day age heuristic.',limitation='A gap overlap is a request to verify missing release timing, not an inferred release. This checks internal quarter sequence completeness; IPO/pre-first-release and unobserved terminal releases remain explicit, and quarter inventories outside candidate exposure are not filled speculatively.')

def main():
 p=ROOT/'outputs/monthly_architecture_restart/META.json';ep=HERE/'events.json';j=audit_events(json.load(open(ep)),json.load(open(p)))
 resolutions=json.load(open(HERE/'GAP_TIMING_RESOLUTIONS.json'));resolved=[];unresolved=[]
 for r in j['internal_gap_reviews']:
  proofs=[x for x in resolutions if x['ticker']==r['ticker'] and dict(fiscal_year=x['fiscal_year'],fiscal_quarter=x['fiscal_quarter']) in r['missing_quarters'] and x['verified_release_date']>r['signal_date']]
  if len(proofs)==len(r['missing_quarters']):
   for x in proofs:assert sha((ROOT/x['evidence_path']).read_bytes())==x['sha256']
   resolved.append(dict(r,resolution='Every missing quarter actually released after this candidate close; no then-known release omitted.',evidence=proofs))
  else:unresolved.append(r)
 j['gap_intersections_resolved_by_actual_release_date']=resolved;j['internal_gap_reviews']=unresolved
 j.update(generated_at=now(),status='read_only_candidate_quarter_sequence_audit',inputs={rel(p):sha(p.read_bytes()),rel(ep):sha(ep.read_bytes())},crm_fy2022q3=dict(release_date='2021-11-30',actual_period_end='2021-10-31',available_session='2021-12-01',candidate_dates=['2021-10-29','2021-11-30'],effect='Neither candidate close has this release available; full original now closes the historical sequence hole.'))
 dump(HERE/'QUARTER_CANDIDATE_AUDIT.json',j);print('Internal gap candidate reviews',len(j['internal_gap_reviews']));print('No prior',[(r['ticker'],r['scope'],r['no_prior_verified_event_dates']) for r in j['per_ticker_scope'] if r['no_prior_verified_event_dates']])
if __name__=='__main__':main()
