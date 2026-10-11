"""Audit archived prices without downloading or modifying raw data or paper accounts."""
from pathlib import Path
import hashlib
import json
import pandas as pd
import exchange_calendars as xc

P = Path(__file__).resolve().parent

def main():
    sessions = xc.get_calendar('XNYS', start='2010-01-01', end='2026-10-05').sessions.tz_localize(None)
    # exchange_calendars 4.5.6 predates Carter closure; official Nasdaq notice ETA2024-87.
    sessions = sessions.difference(pd.DatetimeIndex(['2025-01-09']))
    records, anomalies, hashes = [], [], {}
    current = pd.read_csv(P / 'sources/current_sp500.csv')
    for symbol in current.Symbol.str.replace('.', '-', regex=False):
        path = P / 'raw' / (symbol + '.csv.gz')
        hashes[symbol] = hashlib.sha256(path.read_bytes()).hexdigest()
        d = pd.read_csv(path, parse_dates=['Date']).set_index('Date').sort_index()
        prices = d[['Open', 'High', 'Low', 'Close', 'Adj Close']]
        bad = prices.isna().any(axis=1) | (prices <= 0).any(axis=1)
        bad |= (d.High < d[['Open', 'Low', 'Close']].max(axis=1)) | (d.Low > d[['Open', 'High', 'Close']].min(axis=1))
        bad |= d.Volume.isna() | (d.Volume < 0)
        expected = sessions[(sessions >= d.index.min()) & (sessions <= sessions.max())]
        missing = expected.difference(d.index)
        extra = d.index.difference(sessions)
        recent = sessions[-253:]
        aligned = d.reindex(recent)
        liquid = (aligned.Close * aligned.Volume).tail(20).mean() > 20_000_000
        valid = not aligned[['Open', 'High', 'Low', 'Close', 'Adj Close']].isna().any().any()
        valid = valid and not bad.reindex(recent, fill_value=True).any() and liquid and not d.index.duplicated().any()
        records.append(dict(symbol=symbol, rows=len(d), first=str(d.index.min().date()), last=str(d.index.max().date()), missing_sessions=len(missing), non_session_rows=len(extra), duplicate_dates=int(d.index.duplicated().sum()), invalid_rows=int(bad.sum()), eligible_now=bool(valid)))
        for date in missing:
            anomalies.append(dict(symbol=symbol, date=str(date.date()), reason='missing_session'))
        for date in d.index[bad]:
            anomalies.append(dict(symbol=symbol, date=str(date.date()), reason='invalid_ohlcv'))
        for date in extra:
            anomalies.append(dict(symbol=symbol, date=str(date.date()), reason='non_session'))
    result = pd.DataFrame(records)
    result.to_csv(P / 'calendar_audit.csv', index=False)
    pd.DataFrame(anomalies, columns=['symbol', 'date', 'reason']).to_csv(P / 'quarantine.csv', index=False)
    (P / 'raw_hashes.json').write_text(json.dumps(hashes, indent=2))
    print(result[['rows', 'missing_sessions', 'non_session_rows', 'duplicate_dates', 'invalid_rows', 'eligible_now']].sum().to_json())
    print(result.loc[(result.missing_sessions > 0) | (result.invalid_rows > 0) | ~result.eligible_now].to_string(index=False))

if __name__ == '__main__':
    main()
