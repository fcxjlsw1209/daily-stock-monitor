"""Download missing union candidates; preserve corporate action columns and provenance."""
from pathlib import Path
import datetime
import hashlib
import json
import time
import argparse
import pandas as pd
import yfinance as yf

P = Path(__file__).resolve().parent


def main():
    directory = P / 'sources/yahoo_prices'
    directory.mkdir(exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='+', help='Explicit partial-coverage candidates to supplement')
    args = parser.parse_args()
    symbols = args.symbols or json.loads((P / 'UNION_PRICE_DISCOVERY_AUDIT.json').read_text())['candidates_without_any_prices']
    # Vendor spelling only. Economic aliases require separate verified mappings.
    spellings = {'BRK.B': 'BRK-B'}
    old_path = directory / 'attempts.json'
    previous = {r['symbol']: r for r in json.loads(old_path.read_text())} if old_path.exists() else {}
    rows = list(previous.values())
    def remember(result):
        previous[result['symbol']] = result
        old_path.write_text(json.dumps(list(previous.values()), indent=2)+'\n')
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    for symbol in symbols:
        path = directory / (symbol + '.csv')
        provenance = path.with_suffix('.provenance.json')
        if path.exists() and provenance.exists():
            m = json.loads(provenance.read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest() != m['sha256']:
                raise ValueError('Source cache checksum mismatch')
            rows.append(previous.get(symbol, dict(symbol=symbol, status='cached')))
            continue
        if previous.get(symbol, {}).get('attempt_date') == today:
            rows.append(previous[symbol])
            continue
        result = dict(symbol=symbol, attempt_date=today, identity_verified=False)
        try:
            d = yf.Ticker(spellings.get(symbol, symbol)).history(start='2008-01-01', end='2026-10-07',
                       auto_adjust=False, actions=True, raise_errors=True, timeout=20)
            if d.empty:
                raise ValueError('Empty price response')
            d = d.reset_index()
            d['Date'] = pd.to_datetime(d.Date).dt.strftime('%Y-%m-%d')
            d.to_csv(path, index=False)
            provenance.write_text(json.dumps(dict(provider='Yahoo Finance via yfinance', symbol=spellings.get(symbol,symbol),
                    start='2008-01-01', end_exclusive='2026-10-07', auto_adjust=False,
                    retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), identity_verified=False), indent=2)+'\n')
            result.update(status='downloaded', rows=len(d), first=d.Date.min(), last=d.Date.max())
        except Exception as exc:
            result.update(status='unavailable', error_type=type(exc).__name__)
            if 'RateLimit' in type(exc).__name__:
                rows.append(result)
                remember(result)
                print('Rate limited; stop this source without retries', flush=True)
                return
        rows.append(result)
        remember(result)
        print(symbol, result['status'], result.get('rows',0), flush=True)
        time.sleep(.7)


if __name__ == '__main__':
    main()
