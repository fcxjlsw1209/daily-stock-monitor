"""Offline original-PDF + downstream availability validation; no requests."""
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path
import collect as c


def main():
    events=json.loads((c.HERE/'events.json').read_text())
    original=c.validate(events)
    identity=json.loads((c.HERE/'IDENTITY.json').read_text())
    raw=c.ROOT/identity['evidence_path']
    if c.sha(raw.read_bytes())!=identity['evidence_sha256']:
        raise ValueError('Identity original evidence checksum mismatch')
    itext=' '.join(c.pdf_text(raw).split())
    if not all(x in itext for x in ["June 9, 2022", "'FB'", "'META'", 'CUSIP number will remain unchanged']):
        raise ValueError('Identity continuity and ticker effective date not in original')
    sys.path.insert(0,str(c.HERE.parent))
    from event_schema import normalize_event
    import exchange_calendars as xcals
    cal=xcals.get_calendar('XNYS',start='2012-01-01',end='2027-12-31')
    normalized=[]
    for e in events:
        if e['release_date']>c.CUTOFF:raise ValueError('Future post-cutoff release')
        n=normalize_event({'symbol':e['ticker'],'asset_type':'equity','official':{
            'actual_period_end':e['fiscal_period_end_actual'],'publication_date':e['release_date'],
            'publication_at':None,'publication_timezone':e['publication_timezone'],
            'publication_kind':'release','source_url':e['source_url'],'verified':True}},cal)
        if n['status']!='ready':raise ValueError('Downstream pending: '+str(n))
        if n['available_session']<=e['release_date']:raise ValueError('Date-only event became available too early')
        normalized.append(n)
    c.dump(c.HERE/'normalized_validation.json',normalized)
    status={'validation_at':dt.datetime.now(dt.timezone.utc).isoformat(),
        **original,'normalized_ready':len(normalized),'publication_precision':dict(Counter(e['precision'] for e in events)),
        'historical_ticker_counts':dict(Counter(e['historical_ticker'] for e in events)),
        'all_research_tickers':sorted(set(e['ticker'] for e in events)),
        'release_range':[min(e['release_date'] for e in events),max(e['release_date'] for e in events)],
        'period_range':[min(e['fiscal_period_end_actual'] for e in events),max(e['fiscal_period_end_actual'] for e in events)],
        'releases_since_2015':sum(e['release_date']>='2015-01-01' for e in events),
        'study_window_count':sum(e['in_study_window'] for e in events),
        'events_sha256':c.sha((c.HERE/'events.json').read_bytes()),
        'identity_sha256':identity['evidence_sha256'],
        'limitations':['Date-only original datelines; no first-public exact times verified.',
            'Issuer archive retrieved now, not a contemporaneously captured immutable news feed.',
            'Coverage is quarterly regular earnings sequence, not proof that every unscheduled update/restatement is included.',
            'Use validated META issuer price history; current FB ticker is a different reused security.']}
    c.dump(c.HERE/'VALIDATION.json',status)
    c.dump(c.HERE/'READINESS.json',{'official_date_quarter_coverage_ready':not original['gaps'],
        'conservative_session_availability_ready':len(normalized)==57,
        'exact_publication_timestamp_ready':False,'covered_fiscal_quarters':'2012Q2..2026Q2',
        'cutoff':c.CUTOFF,'gap_count':len(original['gaps']),'identity_verified':True,
        'not_authorized_to_claim':'Exact announcement-time open gap or intraday release reaction',
        'permitted_research_input':'Post-conservative-known-by completed-session price/volume reactions, subject to actual bar coverage and signal cutoff.'})
    print(json.dumps(status,indent=2))

if __name__=='__main__':main()
