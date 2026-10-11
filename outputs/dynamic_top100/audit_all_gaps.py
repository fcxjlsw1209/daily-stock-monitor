"""Read-only source audit; writes separate audit artifacts, never trading data."""
from pathlib import Path
from collections import defaultdict
import json,sqlite3,hashlib,datetime
import pandas as pd
import exchange_calendars as xc
from reconcile_share_evidence import reconcile
from resolve_share_amendments import reviewed_events
P=Path(__file__).resolve().parent
start,end='2010-01-04','2026-10-05'
calendar=xc.get_calendar('XNYS',start=start,end=end)
expected=set(calendar.sessions.strftime('%Y-%m-%d'))
# Official NYSE/Nasdaq closure omitted by installed calendar version.
# https://www.nasdaqtrader.com/TraderNews.aspx?id=ETA2024-87
expected.discard('2025-01-09')
dates=defaultdict(set);source_rows=quarantined=outside=0;failures=[];files=0
for f in sorted((P/'recovered_prices').glob('*.csv.gz')):
 try:
  d=pd.read_csv(f,usecols=['date','symbol','quarantine']);files+=1;source_rows+=len(d)
  q=d.quarantine.astype(str).str.lower().eq('true');quarantined+=int(q.sum())
  d=d[~q & d.date.between(start,end)]
  for symbol,g in d.groupby('symbol'):dates[symbol].update(g.date)
 except Exception as e:failures.append({'file':str(f.relative_to(P)),'error':type(e).__name__})
db_checks={}
for name,dbpath,query in [('expanded',P.parent/'universe_expansion/research_prices.sqlite','SELECT symbol,Date FROM daily_prices WHERE Open>0 AND Close>0 AND Low>0 AND High>=Low AND Open>=Low-0.00001 AND Open<=High+0.00001 AND Close>=Low-0.00001 AND Close<=High+0.00001'),('original',P.parent/'stock_database/stocks.sqlite','SELECT symbol,trade_date FROM daily_prices WHERE open>0 AND close>0 AND low>0 AND high>=low AND open>=low-0.00001 AND open<=high+0.00001 AND close>=low-0.00001 AND close<=high+0.00001')]:
 with sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True) as db:
  db_checks[name]=db.execute('PRAGMA quick_check').fetchone()[0]
  for symbol,day in db.execute(query):
   if start<=day<=end:dates[symbol].add(day)
known=pd.read_csv(P/'CANDIDATE_GAPS.csv').set_index('symbol')
priority={r['symbol']:r for r in json.loads((P/'priority_recovery_summary.json').read_text())['rows']}
sh,_=reviewed_events()
sh.cik=sh.cik.str.zfill(10)
sh=reconcile(sh,json.loads((P/'pilot_identity/share_corrections.json').read_text()))
share_ciks=set(sh.loc[~sh.conflict & sh.shares.gt(0) & ~sh.scale_review_required,'cik'])
ident=pd.read_csv(P/'expanded_sec/candidate_identity_evidence.csv',dtype=str)
share_symbols=set(ident.loc[ident.cik.str.zfill(10).isin(share_ciks),'symbol'])
symbols=sorted(set(known.index)|set(priority))
rows=[]
for s in symbols:
 ds=dates[s]&expected;first=min(ds) if ds else None;last=max(ds) if ds else None
 interior=sorted(x for x in expected if first and first<=x<=last and x not in ds)
 prev=known.loc[s].to_dict() if s in known.index else {};pr=priority.get(s,{})
 rows.append(dict(symbol=s,union_observed_sessions=len(ds),first_observed=first,last_observed=last,interior_sessions_without_any_source=len(interior),first_missing_sessions=';'.join(interior[:12]),raw_share_evidence_present=s in share_symbols or int(prev.get('share_events',0) or 0)>0 or pr.get('share_fact_rows',0)>0,ambiguous_cik=bool(prev.get('ambiguous_cik',False)),end_before_audit_cutoff=bool(last and last<end),identity_verified=False,rank_ready=False))
pd.DataFrame(rows).to_csv(P/'ALL_SYMBOL_GAPS.csv',index=False)
episodes=json.loads((P/'listing_episodes_unverified.json').read_text())
checks=[]
for f in (P/'additional_sources/alpha_vantage').glob('????-??-??_*.json'):
 m=json.loads(f.read_text());c=f.with_suffix('.csv');checks.append(c.exists() and hashlib.sha256(c.read_bytes()).hexdigest()==m.get('sha256'))
issues=[
 ('P0','universe','Confirmed PTR omission in listing vendor; yearly snapshots do not establish complete daily eligible issuers or top100 boundary.'),
 ('P0','identity','No certified effective-dated issuer/security mapping; symbol reuse, mergers, class changes, ADR eligibility and one-security-per-issuer rule unresolved.'),
 ('P0','market_cap','No certified daily issuer market caps: split/ADR/class/FX reconciliation and contemporaneously available shares remain unverified.'),
 ('P0','prices','Source unions are ticker-based discovery only. They may hide identity mismatches, adjustment conflicts and gaps; do not treat union as executable price series.'),
 ('P0','terminal_events','Retired issuers require verified final tradable session and cash/stock distributions with timing; cannot forward-fill or sell at invented close.'),
 ('P0','disclosure_units','Priority SEC sources are normalized, but observation presence does not certify total-company shares, economic class rights or ADR units.'),
 ('P0','daily_membership','No validated real-data daily top100 snapshots or universe completeness certificates have been produced.'),
 ('P1','sector','Historical effective-dated sectors and sector ETF assignment have not been established.'),
 ('P1','backtest','Membership contract and factor engine exist; real-data six-strategy execution and end-to-end corporate-action validation remain incomplete.'),
 ('P1','paper','Runner currently reads migration STATUS only for reporting; live six accounts still use frozen static universe, not dynamic membership.'),
 ('P1','refresh','No operational daily certified dynamic source feed, completeness gating/freeze schedule or new-version migration validation yet.'),
 ('P1','audit_scope','872 known ticker candidates are not the full historical eligible market. Missing unknown issuers and exact top100 boundary cannot yet be quantified.'),
 ('P2','aggregation','Old STATUS totals and coverage files describe different source subsets; summing them can double-count overlapping providers and aliases.'),
 ('P2','validation','Passing code tests validates programmed checks, not economic identity, provider completeness or backtest correctness on real data.')]
report=dict(audit_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),period=[start,end],expected_xnys_sessions=len(expected),scope='All locally known candidate symbols and all recovered price files; not a claim of exhaustive global issuer coverage',candidate_count=len(rows),recovered_files=files,recovered_source_rows=source_rows,recovered_quarantined_rows=quarantined,read_failures=failures,db_quick_checks=db_checks,listing_checksum_checks=len(checks),listing_checksum_failures=checks.count(False),listing_episodes=len(episodes),listing_episodes_without_possible_cik=sum(not e['possible_ciks'] for e in episodes),listing_episodes_with_multiple_possible_ciks=sum(len(e['possible_ciks'])>1 for e in episodes),normalized_share_events=len(sh),normalized_conflicts=int(sh.conflict.sum()),source_verified_share_corrections=int(sh.correction_source.ne('').sum()),unresolved_share_scale_reviews=int(sh.scale_review_required.sum()),known_candidates_without_any_price=[r['symbol'] for r in rows if not r['union_observed_sessions']],known_candidates_without_share_evidence=[r['symbol'] for r in rows if not r['raw_share_evidence_present']],symbols_with_interior_source_gaps=sum(r['interior_sessions_without_any_source']>0 for r in rows),interior_source_gap_sessions=sum(r['interior_sessions_without_any_source'] for r in rows),rank_ready=0,issues=[dict(priority=a,area=b,finding=c) for a,b,c in issues],limitations=['Interior gaps may be legitimate halts, not confirmed missing tradable sessions.','A source date match is not proof of consistent prices or historical company identity.','An early ending series may be a valid delisting; requires lifecycle evidence.','Warmup before 2010 and all listing intervals remain separately required.'])
(P/'ALL_GAPS_AUDIT.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['issues','limitations','known_candidates_without_any_price','known_candidates_without_share_evidence']},indent=2))
print('no prices',len(report['known_candidates_without_any_price']),'no shares',len(report['known_candidates_without_share_evidence']))
