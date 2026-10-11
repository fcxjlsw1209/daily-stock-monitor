"""Lifecycle-aware triage, not a certified price merge or trade input."""
from pathlib import Path
from collections import defaultdict
import json,sqlite3
import pandas as pd
import exchange_calendars as xc
P=Path(__file__).resolve().parent
rules=[('Q_legacy','Q','2013-05-09','2017-11-14'),('IQV_after_rename','IQV','2017-11-15','2026-10-05'),('DOW_legacy','DOW','2010-01-04','2017-08-31'),('DOW_spinoff','DOW','2019-04-02','2026-10-05'),('BHF_regular','BHF','2017-08-07','2026-10-05'),('FOX_legacy','FOX','2013-07-01','2019-03-18'),('FOXA_legacy','FOXA','2013-07-01','2019-03-18'),('FOX_spinoff','FOX','2019-03-19','2026-10-05'),('FOXA_spinoff','FOXA','2019-03-19','2026-10-05'),('SE_legacy','SE','2010-01-04','2017-02-24'),('SE_Sea','SE','2017-10-20','2026-10-05')]
symbols={r[1] for r in rules};dates=defaultdict(set);sources=[]
for f in sorted((P/'recovered_prices').glob('*.csv.gz')):
 d=pd.read_csv(f,usecols=['date','symbol','quarantine']);d=d[d.symbol.isin(symbols)&~d.quarantine.astype(str).str.lower().eq('true')]
 for s,g in d.groupby('symbol'):
  dates[s].update(g.date);sources.append(dict(source=f.name,symbol=s,rows=len(g),first=g.date.min(),last=g.date.max()))
for file,col in [(P.parent/'universe_expansion/research_prices.sqlite','Date'),(P.parent/'stock_database/stocks.sqlite','trade_date')]:
 with sqlite3.connect('file:'+str(file)+'?mode=ro',uri=True) as c:
  for s in symbols:
   records=c.execute('SELECT '+col+' FROM daily_prices WHERE symbol=?',(s,)).fetchall();ds={x[0] for x in records};dates[s].update(ds)
   if ds:sources.append(dict(source=str(file),symbol=s,rows=len(ds),first=min(ds),last=max(ds)))
# Official Nasdaq symbol-reuse mapping, restricted to the old issuer's interval.
# This closes date-presence gaps only, not price-basis or corporate-action certification.
q_validation=json.loads((P/'Q_IQV_RENAME_VALIDATION.json').read_text())
assert q_validation['effective']=='2017-11-15'
assert q_validation['overlap_rows']>=1000 and q_validation['max_relative_difference']<1e-6
q_recovered={r['date'] for r in q_validation['missing_dates_found'] if '2013-05-09'<=r['date']<'2017-11-15'}
# Require a currently present IQV source observation, not just a saved audit claim.
q_recovered &= dates['IQV']
dates['Q'].update(q_recovered)
sources.append(dict(source='Q_IQV_RENAME_VALIDATION.json',symbol='Q',source_symbol='IQV',rows=len(q_recovered),mapping_source=q_validation['source'],price_basis_certified=False))
for alias,legacy,start,end in [('TFCF','FOX','2013-07-01','2019-03-18'),('TFCFA','FOXA','2013-07-01','2019-03-18')]:
 f=P/'recovered_prices'/('tiingo_'+alias+'.csv.gz')
 if f.exists():
  d=pd.read_csv(f);d=d[~d.quarantine.astype(str).str.lower().eq('true') & d.date.between(start,end)]
  dates[legacy].update(d.date)
  sources.append(dict(source=f.name,symbol=legacy,source_symbol=alias,rows=len(d),first=d.date.min(),last=d.date.max(),mapping_source='https://www.nasdaqtrader.com/TraderNews.aspx?id=dtn2019-05'))
cal=set(xc.get_calendar('XNYS',start='2010-01-01',end='2026-10-05').sessions.strftime('%Y-%m-%d'))-{'2025-01-09'}
rows=[]
for name,s,start,end in rules:
 expected={x for x in cal if start<=x<=end};seen=dates[s]&expected;missing=sorted(expected-seen)
 rows.append(dict(episode=name,symbol=s,start=start,end=end,expected_sessions=len(expected),observed_sessions=len(seen),candidate_missing_sessions=len(missing),missing_dates=missing,first_observed=min(seen) if seen else None,last_observed=max(seen) if seen else None,identity_verified=False,interval_status='triage window; IPO/legacy terminal dates require primary verification unless already in VERIFIED_IDENTITY_EVENTS.json',usable_for_backtest=False))
report={'scope':'Targeted triage only; no adjustments or issuer continuity certified. Includes missing episode endpoints, unlike previous interior-only audit. Database dates are presence evidence, not validated OHLC.','episodes':rows,'source_coverage':sources,'excluded_BHF_pre_when_issued_dates':len({x for x in dates['BHF'] if x<'2017-07-17'}),'legacy_Q_after_2017_11_14_must_not_join_new_Q':len({x for x in dates['Q'] if x>'2017-11-14'})}
(P/'IDENTITY_EPISODE_GAPS.json').write_text(json.dumps(report,indent=2))
pd.DataFrame([{k:v for k,v in r.items() if k!='missing_dates'} for r in rows]).to_csv(P/'IDENTITY_EPISODE_GAPS.csv',index=False)
print(pd.DataFrame(rows)[['episode','expected_sessions','observed_sessions','candidate_missing_sessions']].to_string(index=False))
