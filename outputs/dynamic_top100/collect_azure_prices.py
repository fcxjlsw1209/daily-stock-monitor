"""Recover public archived intraday evidence; adjusted bars are NOT raw cap prices."""
from pathlib import Path
import datetime, hashlib, json
import pandas as pd
import requests
import exchange_calendars as xc
from audit_price_archives import audit

P = Path(__file__).resolve().parent
A = P / 'additional_sources/azure_firstrate'
BASE = 'https://finstorage6ef5xpkr7mo3s.blob.core.windows.net/finance/'


def regular_daily(frame, schedule):
    """Start-labelled Eastern bars: exclude pre/postmarket and respect early closes.

    Last intraday trade is a close proxy, not certification of the official auction.
    Missing zero-volume bars are allowed by vendor; retain counts for later audit.
    """
    d = frame.copy()
    d['timestamp'] = pd.to_datetime(d['timestamp']).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    if d.timestamp.duplicated().any():
        raise ValueError('Duplicate intraday timestamps')
    d['date'] = d.timestamp.dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
    hours = schedule.rename(columns={'open': 'market_open', 'close': 'market_close'}).copy()
    hours.index = hours.index.strftime('%Y-%m-%d')
    for field in ['market_open', 'market_close']:
        hours[field] = pd.to_datetime(hours[field], utc=True)
    d = d.join(hours[['market_open', 'market_close']], on='date')
    d = d[(d.timestamp >= d.market_open) & (d.timestamp < d.market_close)].sort_values('timestamp')
    out = d.groupby('date').agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'), volume=('volume', 'sum'), bars=('timestamp', 'size'), first_bar=('timestamp', 'first'), last_bar=('timestamp', 'last')).reset_index()
    out['price_basis'] = 'vendor_split_and_dividend_adjusted'
    out['official_open_close_verified'] = False
    return out


def main():
    schedule = xc.get_calendar('XNYS', start='2005-01-01', end='2022-12-31').schedule
    s = requests.Session()
    results = []
    for row in json.loads((A / 'missing_matches.json').read_text()):
        name, symbol = row['name'], row['symbol']
        dest = A / Path(name).name
        meta = dest.with_suffix('.provenance.json')
        if dest.exists() and meta.exists():
            provenance = json.loads(meta.read_text())
            if hashlib.sha256(dest.read_bytes()).hexdigest() != provenance['sha256']:
                raise ValueError('Cache checksum mismatch')
        else:
            r = s.get(BASE + name, timeout=45)
            r.raise_for_status()
            if len(r.content) != row['bytes']:
                raise ValueError('Archive length changed; refresh inventory first')
            dest.write_bytes(r.content)
            provenance = dict(source=BASE + name, sha256=hashlib.sha256(r.content).hexdigest(), retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), identity_and_adjustments_verified=False)
            meta.write_text(json.dumps(provenance, indent=2))
        raw = pd.read_csv(dest, header=None, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        daily = regular_daily(raw, schedule)
        daily['symbol'] = symbol
        checked = audit(daily, 'azure_firstrate_' + symbol)
        # Archive labels PGN as Progress Energy, but all bars begin in 2014,
        # after its 2012 merger. Do not let ticker reuse erase the real gap.
        checked['identity_conflict'] = symbol == 'PGN'
        checked['quarantine'] = checked.quarantine | checked.identity_conflict
        # Explicitly forbid these adjusted closes from capitalization reconstruction.
        checked['usable_for_market_cap'] = False
        checked.to_csv(P / 'recovered_prices' / ('azure_firstrate_' + symbol + '.csv.gz'), index=False, compression='gzip')
        results.append(dict(symbol=symbol, intraday_rows=len(raw), daily_rows=len(checked), first=checked.date.min(), last=checked.date.max(), quarantined=int(checked.quarantine.sum()), source_sha256=provenance['sha256'], usable_for_market_cap=False, usable_for_backtest=False))
        print(results[-1], flush=True)
    (P / 'AZURE_RECOVERY_AUDIT.json').write_text(json.dumps(dict(source='Public anonymous Azure archive; FirstRate-derived adjusted 30-minute bars', scope='Evidence recovery only; historical identity, adjustment completeness and official close still require reconciliation', results=results), indent=2))


if __name__ == '__main__':
    main()
