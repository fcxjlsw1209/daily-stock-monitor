"""Prioritize missing sessions in provisional primary-derived NDX intervals.

Neither this audit nor a zero missing-date count verifies identity or adjustment
units. Incomplete membership is itself a separate, blocking admission gate.
"""
from pathlib import Path
import gzip
import json
import pandas as pd
from research_calendar import sessions as research_sessions

P = Path(__file__).resolve().parent


def missing_runs(required, observed):
    runs, active = [], []
    for day in required:
        if day not in observed:
            active.append(day)
        elif active:
            runs.append(dict(first=active[0], last=active[-1], sessions=len(active)))
            active = []
    if active:
        runs.append(dict(first=active[0], last=active[-1], sessions=len(active)))
    return runs


def main():
    dates = {s:set(ds) for s,ds in json.loads(gzip.decompress((P/'OBSERVED_PRICE_DATES.json.gz').read_bytes())).items()}
    intervals = json.loads((P/'NDX_PRIMARY_CHAIN_AUDIT.json').read_text())['provisional_intervals']
    sessions = list(research_sessions('2008-01-01','2026-10-07').strftime('%Y-%m-%d'))
    rows = []
    for interval in intervals:
        start, end = interval['start'], interval['end_exclusive']
        active = [d for d in sessions if start <= d < end]
        prior = [d for d in sessions if d < start][-252:]
        observed = dates.get(interval['symbol'],set())
        rows.append(dict(**interval, expected_member_sessions=len(active),
                         missing_member_sessions=sum(d not in observed for d in active),
                         missing_warmup_sessions=sum(d not in observed for d in prior),
                         missing_member_runs=missing_runs(active, observed)))
    result = dict(scope='Provisional NDX intervals, same ticker only; SP100 and canonical aliases pending',
                  intervals=rows, intervals_with_missing_member_prices=sum(r['missing_member_sessions']>0 for r in rows),
                  total_missing_member_sessions=sum(r['missing_member_sessions'] for r in rows),
                  admission=False, note='These are raw observed gaps, not certified economic coverage gaps. IPO warmup and renames require explicit resolution.')
    (P/'REQUIRED_INTERVAL_DISCOVERY_AUDIT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='intervals'},indent=2))
    for r in sorted(rows,key=lambda r:r['missing_member_sessions'],reverse=True)[:20]:
        print(r['symbol'],r['start'],r['end_exclusive'],r['missing_member_sessions'])


if __name__ == '__main__':
    main()
