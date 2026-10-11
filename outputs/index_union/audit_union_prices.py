"""Union-specific discovery audit. Ticker coverage is NOT security certification."""
from collections import defaultdict
from pathlib import Path
import datetime
import json
import sqlite3
import sys

import pandas as pd
import exchange_calendars as xc

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P / 'vendor'))
import yaml


def main():
    ndx = yaml.load((P / 'sources/n100tickers/n100-ticker-changes-2010.yaml').read_text(), Loader=yaml.BaseLoader)
    candidates = set(ndx['tickers_on_Jan_1'])
    events = json.loads((P / 'ndx_events_UNVERIFIED.json').read_text())
    for e in events:
        if e['effective_date'] >= '2010-01-01':
            candidates.update(e['add'])
            candidates.update(e['remove'])
    corrob = json.loads((P / 'sp100_2009_holdings_corrob.json').read_text())
    candidates.update(corrob['tickers_inferred_from_printed_company_names'])
    candidates.update(['MET', 'NWSA', 'FCX', 'BRK.B'])
    for name in ['sp100_legacy_reviewed_rows.json', 'sp100_modern_event_rows.json', 'sp100_supplemental_reviewed_rows.json']:
        for row in json.loads((P / name).read_text()):
            candidates.add(row.get('ticker_as_announced_or_effective', row.get('ticker')))
    for row in json.loads((P / 'ndx_primary_reviewed_rows.json').read_text()):
        if row['effective_session'] >= '2010-01-01':
            candidates.add(row['ticker_as_announced_or_effective'])
    for f in sorted(P.glob('SECURITY_ALIAS_EVIDENCE*.json')):
        for alias in json.loads(f.read_text())['aliases']:
            candidates.update([alias['old'], alias['new']])
    candidates.discard(None)
    observed = defaultdict(set)
    errors = []
    for f in sorted((P.parent / 'dynamic_top100/recovered_prices').glob('*.csv.gz')):
        try:
            d = pd.read_csv(f, usecols=['date', 'symbol', 'quarantine'])
            d = d[d.symbol.isin(candidates) & ~d.quarantine.astype(str).str.lower().eq('true')]
            for symbol, g in d.groupby('symbol'):
                observed[symbol].update(g.date)
        except Exception as exc:
            errors.append(dict(file=str(f), error=type(exc).__name__))
    databases = [('universe_expansion/research_prices.sqlite', 'Date'), ('stock_database/stocks.sqlite', 'trade_date')]
    for file, date_column in databases:
        with sqlite3.connect('file:' + str(P.parent / file) + '?mode=ro', uri=True) as db:
            for symbol, day in db.execute('SELECT symbol, ' + date_column + ' FROM daily_prices WHERE open>0 AND close>0 AND low>0 AND high>=low AND open>=low-0.00001 AND open<=high+0.00001 AND close>=low-0.00001 AND close<=high+0.00001'):
                if symbol in candidates:
                    observed[symbol].add(day)
    new_inventory = P / 'NEW_PRICE_VALIDATION.json'
    if new_inventory.exists():
        for item in json.loads(new_inventory.read_text())['files']:
            f = P / item['normalized_file']
            import hashlib
            if hashlib.sha256(f.read_bytes()).hexdigest() != item['normalized_sha256']:
                raise ValueError('Normalized cache checksum mismatch: ' + str(f))
            d = pd.read_csv(f, usecols=['date','symbol','bar_valid','quarantine_reason'])
            d = d[d.bar_valid & d.quarantine_reason.isna()]
            for symbol, g in d.groupby('symbol'):
                if symbol in candidates:
                    observed[symbol].update(g.date)
    rows = []
    for symbol in sorted(candidates):
        ds = observed[symbol]
        rows.append(dict(symbol=symbol, observed_dates=len(ds), first=min(ds) if ds else None,
                         last=max(ds) if ds else None, identity_verified=False))
    result = dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  candidate_count=len(rows), candidates_without_any_prices=[r['symbol'] for r in rows if not r['observed_dates']],
                  rows=rows, read_errors=errors, admission=False,
                  limitations=['Superset discovery candidates; not certified membership.',
                               'Ticker reuse and renamed symbols are not merged.',
                               'Observed dates do not certify daily continuity, OHLCV units or adjustments.'])
    (P / 'UNION_PRICE_DISCOVERY_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    # Keep dates for interval-based diagnostics; a same-ticker hit is never admission.
    import gzip
    with gzip.GzipFile(str(P/'OBSERVED_PRICE_DATES.json.gz'), 'wb', mtime=0) as out:
        out.write(json.dumps({s: sorted(ds) for s, ds in sorted(observed.items())}, sort_keys=True).encode())
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
