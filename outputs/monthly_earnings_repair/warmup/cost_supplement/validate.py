"""Offline evidence verification and candidate-window audit; no portfolio writes."""
from collect import *
from collections import Counter

def validate(e):
 assert e['ticker']=='COST' and e['precision']=='date_only' and e['publication_timestamp'] is None
 assert e['publication_timezone']=='America/Los_Angeles' and not e['first_public_time_verified']
 assert e['fiscal_period_end_actual']<e['release_date']<=CUTOFF
 for proof in [dict(evidence_path=e['evidence_path'],sha256=e['evidence_sha256'])]+e['release_date_evidence']+[e['fiscal_period_evidence'],e['fiscal_identity_evidence']]+e.get('additional_evidence',[]):
  p=ROOT/proof['evidence_path'];assert p.is_relative_to(HERE);assert sha(p.read_bytes())==proof['sha256'],p
 n=normalize_event(dict(symbol='COST',official=dict(verified=True,publication_kind='release',publication_date=e['release_date'],publication_timezone=e['publication_timezone'],actual_period_end=e['fiscal_period_end_actual'],source_url=e['source_url'])))
 assert n['status']=='ready' and not n['known_by_is_actual_publication'];assert n['available_session']>e['release_date']
 return n

def main():
 es=json.load(open(HERE/'events.json'));all_es=json.load(open(HERE/'ALL_VERIFIED_CANDIDATES.json'));aud=json.load(open(HERE/'ORIGINAL_35_AUDIT.json'));cor=json.load(open(HERE/'CORRECTIONS.json'))
 for e in all_es:validate(e)
 assert len(all_es)==len({(e['fiscal_year'],e['fiscal_quarter']) for e in all_es})
 obs={(e['fiscal_year'],e['fiscal_quarter']) for e in all_es};assert all((y,q) in obs for y in range(2018,2027) for q in range(1,5))
 assert len(aud)==35 and all(r['status']!='independent_reverification_pending' for r in aud)
 source=ROOT/'outputs/monthly_architecture_restart/META.json';meta=json.load(open(source));rows=[]
 normalized=[(e,validate(e)) for e in all_es]
 for scope in ['monthly','training']:
  for d,syms in zip(meta[scope]['dates'],meta[scope]['symbols']):
   if 'COST' not in syms:continue
   opts=[(e,n) for e,n in normalized if n['available_session']<=d];latest=max(opts,key=lambda x:x[0]['release_date']) if opts else None
   age=(dt.date.fromisoformat(d)-dt.date.fromisoformat(latest[0]['release_date'])).days if latest else None
   rows.append(dict(scope=scope,signal_date=d,prior_release=latest[0]['release_date'] if latest else None,age_days=age,prior_fy=latest[0]['fiscal_year'] if latest else None,prior_q=latest[0]['fiscal_quarter'] if latest else None,state='no_prior' if latest is None else 'over_120d' if age>120 else 'within_120d'))
 summary={scope:dict(rows=sum(r['scope']==scope for r in rows),review_rows=sum(r['scope']==scope and r['state']!='within_120d' for r in rows)) for scope in ['monthly','training']}
 dump(HERE/'CANDIDATE_AUDIT.json',dict(inputs={rel(source):sha(source.read_bytes())},counts=summary,rows=rows,continuous_fiscal_quarters='FY2018Q1 through FY2026Q4 inclusive; no missing quarter identity',limitation='Age threshold is a secondary diagnostic only. Sequence completeness is independently checked over the entire candidate exposure plus its predecessor.'))
 missing=[dict(fiscal_year=y,fiscal_quarter=q) for y in range(2011,2027) for q in range(1,5) if (y,q) not in obs]
 dump(HERE/'STATUS.json',dict(status='verified_candidate_windows_ready_to_merge',updated_at=now(),new_event_count=len(es),independent_candidates=len(all_es),original_events_audited=len(aud),confirmed_corrections=len(cor['corrections']),source_classes=dict(Counter(e['source_class'] for e in es)),continuous_quarter_coverage='FY2018Q1-FY2026Q4',candidate_counts=summary,events_sha256=sha((HERE/'events.json').read_bytes()),unverified_historical_inventory=missing,original_snapshot=cor['original_input'],original_snapshot_sha256=cor['original_input_sha256'],integration_policy='Merge 19 additions and apply the single CORRECTIONS overlay explicitly; do not replace the original immutable source.'))
 dump(HERE/'VALIDATION.json',dict(status='passed',validated_at=now(),new_events_checked=len(es),all_independent_events_checked=len(all_es),old_events_audited=len(aud),checks=['all primary and secondary evidence SHA-256 checks','central schema normalization and date-only next-session availability','FY2018Q1-FY2026Q4 inclusive quarter continuity','explicit 12/16/17-week period anchors, separate from monthly sales','immutable financial687 source audited without modification'],network_used=False))
 print('COST validation',len(es),'new',len(all_es),'total;',summary,'unverified inventory',missing)
if __name__=='__main__':main()
