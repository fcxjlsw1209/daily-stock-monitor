"""Compare primary reviewed events with independent discovery; never auto-certify."""
from pathlib import Path
import json
import hashlib
from collections import defaultdict

P = Path(__file__).resolve().parent


def main():
    discovery = json.loads((P/'ndx_events_UNVERIFIED.json').read_text())
    reviewed = json.loads((P/'ndx_primary_reviewed_rows.json').read_text())
    primary = defaultdict(lambda: {'add': set(), 'remove': set()})
    for row in reviewed:
        f=P/row['source_file']
        assert hashlib.sha256(f.read_bytes()).hexdigest() == row['source_sha256'], str(f)
        primary[row['effective_session']]['add' if row['action']=='addition' else 'remove'].add(row['ticker_as_announced_or_effective'])
    hints = defaultdict(lambda: {'add': set(), 'remove': set()})
    for row in discovery:
        for field in ['add','remove']:
            hints[row['effective_date']][field].update(row[field])
    rows=[]
    for day in sorted(set(hints)|set(primary)):
        if day < '2010-01-01' or day > '2026-10-06': continue
        a,b=hints[day],primary[day]
        rows.append(dict(date=day,exact_match=a==b,
            missing_primary_add=sorted(a['add']-b['add']), missing_primary_remove=sorted(a['remove']-b['remove']),
            primary_extra_add=sorted(b['add']-a['add']), primary_extra_remove=sorted(b['remove']-a['remove'])))
    result=dict(reviewed_rows=len(reviewed), discovery_dates=len(hints),
                exact_matched_dates=sum(r['exact_match'] for r in rows), reviewed_dates=rows,
                identity_and_aliases_pending=True, complete_history_certified=False)
    (P/'NDX_PRIMARY_RECONCILIATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Reviewed rows:',len(reviewed),'Exact dates:',result['exact_matched_dates'])
    for r in rows:
        if not r['exact_match']: print(json.dumps(r))


if __name__=='__main__': main()
