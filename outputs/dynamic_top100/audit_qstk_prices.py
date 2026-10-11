"""Quarantine malformed archived bars; discovery evidence is never certified input."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import exchange_calendars as xc

P = Path(__file__).resolve().parent


def clean(raw, sessions, quarantine_from=None):
    d = raw.rename(columns=lambda c: c.strip().lower().replace(' ', '_')).copy()
    values = d[['open', 'high', 'low', 'close', 'volume', 'adj_close']].apply(pd.to_numeric, errors='coerce')
    d[values.columns] = values
    valid = np.isfinite(values).all(axis=1) & values[['open', 'high', 'low', 'close', 'adj_close']].gt(0).all(axis=1)
    valid &= values.volume.ge(0) & values.high.ge(values.low)
    valid &= values.open.between(values.low-1e-5, values.high+1e-5)
    valid &= values.close.between(values.low-1e-5, values.high+1e-5)
    d['quarantine_reason'] = ''
    d.loc[~valid, 'quarantine_reason'] += 'missing_or_invalid_bar;'
    d.loc[~d.date.isin(sessions), 'quarantine_reason'] += 'non_session;'
    d.loc[d.date.duplicated(keep=False), 'quarantine_reason'] += 'duplicate_date;'
    if quarantine_from:
        d.loc[d.date.ge(quarantine_from), 'quarantine_reason'] += 'terminal_date_requires_review;'
    d['quarantine'] = d.quarantine_reason.ne('')
    d['identity_and_adjustments_verified'] = False
    return d


def main():
    root = P/'additional_sources/free_discovery/qstk'
    sessions = set(xc.get_calendar('XNYS', start='2000-01-01', end='2013-01-01').sessions.strftime('%Y-%m-%d'))
    terminals = {r['symbol']:r for r in json.loads((P/'pilot_identity/VERIFIED_TERMINAL_EVIDENCE.json').read_text())}
    rows = []
    for f in sorted(root.glob('*.csv')):
        meta = json.loads(f.with_suffix('.csv.provenance.json').read_text())
        if hashlib.sha256(f.read_bytes()).hexdigest() != meta['sha256']:
            raise ValueError('Archive checksum changed')
        d = clean(pd.read_csv(f), sessions, terminals.get(f.stem, {}).get('quarantine_from'))
        d['symbol'] = f.stem
        d['source'] = meta['source']
        d.to_csv(P/'recovered_prices'/f'qstk_{f.stem}.csv.gz', index=False)
        valid = d[~d.quarantine & d.date.ge('2009-01-01')]
        interior = sorted(x for x in sessions if not valid.empty and valid.date.min()<=x<=valid.date.max() and x not in set(valid.date))
        rows.append(dict(symbol=f.stem, total_rows=len(d), quarantined_rows=int(d.quarantine.sum()),
                         valid_rows_since_2009=len(valid), first=valid.date.min() if len(valid) else None,
                         last=valid.date.max() if len(valid) else None, interior_gap_sessions=len(interior),
                         first_gaps=interior[:10], sha256=meta['sha256']))
    report = dict(source='QuantSoftware/QuantSoftwareToolkit pinned archive', results=rows,
                  valid_rows_since_2009=sum(r['valid_rows_since_2009'] for r in rows),
                  quarantined_rows=sum(r['quarantined_rows'] for r in rows), usable_for_backtest=False,
                  note='Source-level validation only. BTU/EK/RSH are original vendor symbols, not automatically remapped to bankrupt successors. Terminal events and adjustments remain separate.')
    (P/'QSTK_RECOVERY_AUDIT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
