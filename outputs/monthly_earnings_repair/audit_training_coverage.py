"""Prioritize evidence retrieval across both unchanged monthly input panels.

The 120-day review flag is not a completeness test or eligibility change:
annual/year-end calendars and foreign reporting frequencies need evidence review.
"""
from collections import Counter,defaultdict
import hashlib,json
from pathlib import Path
import pandas as pd
from research_calendar import market_calendar
P=Path(__file__).resolve().parent
def main():
    f=P.parent/'monthly_architecture_restart/META.json';meta=json.loads(f.read_text())
    e=P/'EVENTS.json';events=json.loads(e.read_text());groups=defaultdict(list)
    for row in events:
        if row['status']=='ready':groups[row['symbol']].append(row)
    etfs=set(meta['monthly']['etfs']);cal=market_calendar(start='2010-01-01',end='2030-12-31')
    allrows=[];counts={}
    for scope in ['monthly','training']:
        rows=[]
        for day,symbols in zip(meta[scope]['dates'],meta[scope]['symbols']):
            asof=cal.session_close(pd.Timestamp(day))
            for symbol in symbols:
                if symbol in etfs:continue
                known=[x for x in groups[symbol] if pd.Timestamp(x['known_by_at'])<=asof]
                latest=max(known,key=lambda x:x['known_by_at']) if known else None
                age=(asof-pd.Timestamp(latest['known_by_at'])).days if latest else None
                state='no_prior_verified_event' if latest is None else ('review_reporting_interval_over_120d' if age>120 else 'prior_event_within_120d')
                rows.append(dict(scope=scope,signal_date=day,symbol=symbol,state=state,prior_release=latest['publication_date'] if latest else None,age_days=age))
        allrows+=rows;counts[scope]=dict(candidate_rows=len(rows),states=dict(Counter(x['state'] for x in rows)),review_by_symbol=dict(Counter(x['symbol'] for x in rows if x['state']!='prior_event_within_120d')))
    result=dict(status='research_evidence_priority_not_backtest_gate',inputs={str(f):hashlib.sha256(f.read_bytes()).hexdigest(),str(e):hashlib.sha256(e.read_bytes()).hexdigest()},counts=counts,
        caveat='Recent event presence does not certify all releases; >120d can reflect valid annual/half-year calendars. Neither changes candidate eligibility.',review_rows=[x for x in allrows if x['state']!='prior_event_within_120d'])
    (P/'CANDIDATE_HISTORY_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(counts))
if __name__=='__main__':main()
