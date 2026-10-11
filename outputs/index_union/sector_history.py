"""Time-aware sector panel from reviewed effective-dated observations.

No current-sector backfill. A late publication can only affect later signals.
Rows require permanent security IDs and an explicit validated coverage interval.
"""
import pandas as pd
from datetime import datetime,time
from zoneinfo import ZoneInfo


def asof_sectors(rows,sessions,securities):
    result=pd.DataFrame(None,index=sessions,columns=securities,dtype=object)
    ny=ZoneInfo('America/New_York')
    grouped={s:[] for s in securities}
    for r in rows:
        if r['security_id'] not in grouped:continue
        if not r.get('verified') or not r.get('source_sha256'):
            raise ValueError('Sector input lacks reviewed source')
        known=pd.Timestamp(r['known_at'])
        if known.tzinfo is None:raise ValueError('Timezone-aware sector publication required')
        if r['effective_from']>=r['effective_to'] or not r['sector']:
            raise ValueError('Invalid classification interval')
        grouped[r['security_id']].append((r,known))
    for day in sessions:
        label=str(day.date())
        cutoff=pd.Timestamp(datetime.combine(day.date(),time(16),tzinfo=ny))
        for security,records in grouped.items():
            applicable=[r for r,known in records if known<=cutoff and r['effective_from']<=label<r['effective_to']]
            if len(applicable)>1:raise ValueError('Conflicting sector evidence: '+security+' '+label)
            if applicable:result.at[day,security]=applicable[0]['sector']
    return result


def benchmark_returns(sectors,etf_adjusted,mapping):
    if not sectors.index.equals(etf_adjusted.index):raise ValueError('ETF calendar alignment required')
    returns=etf_adjusted.pct_change(60,fill_method=None)
    out=pd.DataFrame(float('nan'),index=sectors.index,columns=sectors.columns)
    for sector,etf in mapping.items():
        if etf not in returns:continue
        for security in sectors:
            mask=sectors[security].eq(sector)
            out.loc[mask,security]=returns.loc[mask,etf]
    return out
