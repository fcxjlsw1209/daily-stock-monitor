"""Offline aggregation and proof validation. Writes only this batch directory."""
from common import *
import sys,collections
sys.path.insert(0,str(HERE.parent))
from event_schema import normalize_event
# These ranges are quarter inventories to audit, not inferred release dates.
RANGES={k:((2010,4),(2026,2)) for k in ['TSLA','INTC','NFLX','AMD','CAT']}
RANGES.update({k:((2011,2),(2026,4)) for k in ['CSCO','LRCX','KLAC']})
RANGES.update({k:((2011,1),(2026,3)) for k in ['ADBE','AVGO','AMAT']})
RANGES.update(CRM=((2011,4),(2027,2)),ACN=((2011,2),(2026,3)),MU=((2011,2),(2026,4)),PLTR=((2020,3),(2026,2)),TMUS=((2013,1),(2026,2)),ARM=((2024,2),(2027,1)),PDD=((2018,2),(2026,2)),SKHY=((2026,2),(2026,2)),SPCX=((2026,2),(2026,2)))
RANGES.update(IBM=((2010,4),(2016,2)),HPQ=((2010,4),(2011,1)),QCOM=((2011,1),(2014,2)))
RANGES['CSCO']=((2011,1),(2026,4))
def quarters(a,b):return [(n//4,n%4+1) for n in range(a[0]*4+a[1]-1,b[0]*4+b[1])]
def validate_event(e):
 assert e['ticker'] in CONFIG
 assert ('2010-10-01' if e.get('training_predecessor_only') else '2011-01-01')<=e['release_date']<=CUTOFF
 assert e['publication_timestamp'] is None and e['precision']=='date_only'
 assert e['first_public_time_verified'] is False
 assert e['fiscal_period_end_actual']<e['release_date']
 proofs=[dict(evidence_path=e['evidence_path'],sha256=e['evidence_sha256'])]+e['release_date_evidence']+[e['fiscal_period_evidence'],e['fiscal_identity_evidence']]+e.get('additional_evidence',[])
 for proof in proofs:
  p=ROOT/proof['evidence_path'];assert p.is_relative_to(HERE),p
  assert p.is_file() and sha(p.read_bytes())==proof['sha256'],p
 n=normalize_event(dict(symbol=e['ticker'],official=dict(verified=True,publication_kind='release',publication_date=e['release_date'],publication_timezone=e['publication_timezone'],actual_period_end=e['fiscal_period_end_actual'],source_url=e['source_url'])))
 assert n['status']=='ready',n
 assert n['known_by_is_actual_publication'] is False
 return n

def finalize():
 events=[];coverage=[];keys=set()
 for k in CONFIG:
  p=HERE/k/'events.json'
  if not p.exists():dump(p,[])
  es=json.loads(p.read_text());observed=set()
  for e in es:
   validate_event(e);key=k,e['fiscal_year'],e['fiscal_quarter'];assert key not in keys,key;keys.add(key);observed.add(key[1:]);events.append(e)
  missing=[dict(fiscal_year=y,fiscal_quarter=q) for y,q in quarters(*RANGES[k]) if (y,q) not in observed]
  coverage.append(dict(ticker=k,coverage_scope=('training_candidate_history_only' if k in ['IBM','HPQ','QCOM'] else 'monthly_and_training'),event_count=len(es),first_release=min((e['release_date'] for e in es),default=None),last_release=max((e['release_date'] for e in es),default=None),source_classes=dict(collections.Counter(e['source_class'] for e in es)),unverified_quarter_inventory=missing,inventory_limitation='Quarter-range inventory only; absent rows are not fabricated release dates. IPO/ADR eligibility windows apply. MU FY2026Q4 cutoff eligibility remains to verify.'))
 events.sort(key=lambda e:(e['ticker'],e['release_date']))
 dump(HERE/'events.json',events)
 status=dict(updated_at=now(),status='partial_verified_ready_to_merge',target_publication_window=['2011-01-01',CUTOFF],training_predecessor_extension='HPQ 2010Q4 and CSCO FY2011Q1 original 2010 releases admitted only as explicitly marked first-training-month predecessors',event_count=len(events),per_ticker=coverage,limitations=['Date-only actual release evidence: publication_timestamp=null; issuer-local end-of-day and next XNYS close policy is applied by the central schema.','Public-index evidence is labelled separately and its SHA identifies the saved observation, not original bytes.','Quarter coverage is incomplete; candidate-month audit and exact unverified inventories remain explicit.','SPCX is the 2026 SpaceX issuer, not the previous ticker-holder fund; SKHY is the 2026 Nasdaq ADR. No pre-listing ticker backfill.'])
 dump(HERE/'STATUS.json',status)
 dump(HERE/'VALIDATION.json',dict(validated_at=now(),status='passed',events_checked=len(events),checks=['every primary and secondary evidence file exists and its SHA-256 matches','unique ticker/fiscal-year/quarter','all records normalize as ready under central event_schema','date-only and null publication timestamps preserved','actual period precedes publication and release is within cutoff','all evidence resides in assigned batch directory'],network_used=False))
 dump(HERE/'PENDING.json',dict(status='incomplete',per_ticker=[dict(ticker=r['ticker'],unverified_quarter_inventory=r['unverified_quarter_inventory'],detail_files=[rel(p) for p in (HERE/r['ticker']).glob('*PENDING.json')]) for r in coverage],source_stops=rel(HERE/'SOURCE_STOPS.json'),frozen_window_audit=rel(HERE/'FROZEN_WINDOW_AUDIT.json')))
 print('Validated aggregate:',len(events));print('\n'.join(f"{r['ticker']}: {r['event_count']} events, {len(r['unverified_quarter_inventory'])} unverified inventory quarters" for r in coverage))
if __name__=='__main__':finalize()
