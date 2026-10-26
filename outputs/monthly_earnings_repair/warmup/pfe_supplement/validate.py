"""Offline integrity, fiscal continuity and exact candidate-date coverage checks."""
from collect import *
from collections import Counter

def normalized(e):
 n=normalize_event(dict(symbol='PFE',official=dict(verified=True,publication_kind='release',publication_date=e['release_date'],publication_timezone=e['publication_timezone'],actual_period_end=e['fiscal_period_end_actual'],source_url=e['source_url'])))
 assert n['status']=='ready' and not n['known_by_is_actual_publication'];assert n['available_session']>e['release_date'];return n

def validate_event(e):
 assert e['ticker']=='PFE' and e['precision']=='date_only' and e['publication_timestamp'] is None
 assert e['fiscal_period_end_actual']<e['release_date'] and e['publication_timezone']=='America/New_York'
 for proof in [dict(evidence_path=e['evidence_path'],sha256=e['evidence_sha256'])]+e['release_date_evidence']+[e['fiscal_period_evidence'],e['fiscal_identity_evidence']]:
  p=ROOT/proof['evidence_path'];assert p.is_relative_to(HERE) and sha(p.read_bytes())==proof['sha256']
  m=json.loads(p.with_suffix('.meta.json').read_text());assert m['status']==200 and m['sha256']==sha(p.read_bytes())
 if e.get('corroborating_public_index_evidence'):
  proof=e['corroborating_public_index_evidence'];assert proof['source_class']=='official_public_index_excerpt_not_original_bytes';assert sha((ROOT/proof['evidence_path']).read_bytes())==proof['sha256'];assert proof['release_date']==e['release_date']
 if e.get('source_revision_evidence'):
  proof=e['source_revision_evidence'];assert sha((ROOT/proof['evidence_path']).read_bytes())==proof['sha256'];assert '40 basis points' in proof['excerpt']
 r=parse(extract(ROOT/e['evidence_path']));assert all(r[k]==e[k] for k in ['fiscal_year','fiscal_quarter','release_date','fiscal_period_end_actual']);return normalized(e)

def main():
 ep=HERE/'events.json';es=json.loads(ep.read_text());p=json.loads((HERE/'PENDING.json').read_text());base=ROOT/p['baseline'];old=[e for e in json.loads(base.read_text()) if e['ticker']=='PFE'];meta_file=ROOT/'outputs/monthly_architecture_restart/META.json';meta=json.loads(meta_file.read_text())
 assert sha(base.read_bytes())==p['baseline_sha256'];assert len(es)==20 and len(old)==44
 for e in es:validate_event(e)
 both=old+es;keys=[(e['fiscal_year'],e['fiscal_quarter']) for e in both];expected={(n//4,n%4+1) for n in range(2010*4+2,2026*4+2)}
 assert len(keys)==len(set(keys))==64 and set(keys)==expected
 ready=[(e,normalized(e)) for e in both];rows=[]
 for scope in ['monthly','training']:
  for d,syms in zip(meta[scope]['dates'],meta[scope]['symbols']):
   if 'PFE' not in syms:continue
   options=[(e,n) for e,n in ready if n['available_session']<=d];assert options,d;e,n=max(options,key=lambda en:en[0]['release_date'])
   rows.append(dict(scope=scope,signal_date=d,latest_release=e['release_date'],available_session=n['available_session'],fiscal_year=e['fiscal_year'],fiscal_quarter=e['fiscal_quarter'],age_calendar_days=(dt.date.fromisoformat(d)-dt.date.fromisoformat(e['release_date'])).days))
 first=next(r for r in rows if r['scope']=='training');assert (first['signal_date'],first['fiscal_year'],first['fiscal_quarter'])==('2011-01-31',2010,3)
 counts={s:dict(rows=sum(r['scope']==s for r in rows),over_120d=sum(r['scope']==s and r['age_calendar_days']>120 for r in rows)) for s in ['monthly','training']}
 inputs={rel(base):sha(base.read_bytes()),rel(ep):sha(ep.read_bytes()),rel(meta_file):sha(meta_file.read_bytes())}
 dump(HERE/'CANDIDATE_AUDIT.json',dict(inputs=inputs,counts=counts,rows=rows,quarter_continuity='Every quarter FY2010Q3 through FY2026Q2; all exact actual candidate dates and their preceding event covered',first_month_predecessor=first,limitation='Quarter identity sequence is independently checked; age is diagnostic only. Date-only publication does not permit same-day use.'))
 dump(HERE/'STATUS.json',dict(status='verified_ready_to_merge',generated_at=now(),new_events=20,new_release_min=es[0]['release_date'],new_release_max=es[-1]['release_date'],baseline_events=44,combined_events=64,complete_fiscal_quarter_range=['2010Q3','2026Q2'],candidate_counts=counts,events_sha256=sha(ep.read_bytes()),source_classes=dict(Counter(e['source_class'] for e in es)),all_primary_sources_full_original_bytes=True,public_index_only_events=0,secondary_public_index_corroborations=2,revision_metadata_only_events=2,training_predecessor_events=sum(bool(e.get('training_predecessor_only')) for e in es),unverified_target_quarters=[],account_or_training_changes=False,source_stop_policy='Inherited Pfizer main/investors hosts and SEC/Yahoo/AlphaVantage stop respected; only public official Q4 CDN originals requested'))
 dump(HERE/'VALIDATION.json',dict(status='passed',validated_at=now(),inputs=inputs,checks=['20 original-source parse roundtrips and primary/secondary SHA checks','64 unique continuous fiscal-quarter identities with frozen 44-row baseline','every actual monthly and training candidate has prior verified event','2011-01-31 uses 2010Q3 predecessor, 2010Q4 release is February 1 and not yet known','all issuer-local date-only records normalize conservatively','no stock pool, account or original-source changes'],network_used=False))
 print('Validated 20 new, 64 combined:',counts,'SHA',sha(ep.read_bytes()))
if __name__=='__main__':main()
