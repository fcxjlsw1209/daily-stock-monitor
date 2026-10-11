"""Read-only Yahoo feasibility probes, with immutable public response caches.

This does not fit models, alter observers, or certify a historical universe.
Endpoint fields verified against upstream yfinance scrapers/quote.py.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from curl_cffi import requests

P = Path(__file__).resolve().parent
CACHE = P / 'raw'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class SourceThrottled(RuntimeError):
    pass


class GuardedSession(requests.Session):
    def request(self, *args, **kwargs):
        state = P / 'RATE_LIMIT.json'
        now = datetime.now(timezone.utc)
        if state.exists():
            z = json.loads(state.read_text())
            if z['utc_date'] == now.date().isoformat():
                raise SourceThrottled('Yahoo stopped for the day after HTTP429')
        response = super().request(*args, **kwargs)
        if response.status_code == 429:
            dump(state, dict(utc_date=now.date().isoformat(),
                             retry_after=response.headers.get('Retry-After'),
                             time_utc=now.isoformat(), source='Yahoo'))
            raise SourceThrottled('HTTP429; no more Yahoo requests today')
        return response


def fetch_cap(symbol, end=None):
    path = CACHE / (symbol + ('_before_' + end if end else '') + '_marketcap.json')
    if path.exists():
        return json.loads(path.read_text())
    params = dict(symbol=symbol,
                  type='monthlyMarketCap,quarterlyMarketCap,trailingMarketCap',
                  period1=int(pd.Timestamp('2015-01-01', tz='UTC').timestamp()),
                  period2=int((pd.Timestamp(end, tz='UTC') if end else pd.Timestamp.now('UTC').ceil('D')).timestamp()))
    url = ('https://query2.finance.yahoo.com/ws/fundamentals-timeseries/'
           'v1/finance/timeseries/' + symbol)
    with GuardedSession(impersonate='chrome') as session:
        response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    value = response.json()
    dump(path, value)
    dump(path.with_suffix('.source.json'), dict(
        url=url, params=params, retrieved_utc=datetime.now(timezone.utc).isoformat(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        historical_publication_vintage_verified=False))
    return value


def fetch_shares(symbol):
    path = CACHE / (symbol + '_shares.json')
    if path.exists():
        value = json.loads(path.read_text())
    else:
        params = dict(symbol=symbol,
            period1=int(pd.Timestamp('2015-01-01', tz='UTC').timestamp()),
            period2=int(pd.Timestamp.now('UTC').ceil('D').timestamp()))
        url = ('https://query2.finance.yahoo.com/ws/fundamentals-timeseries/'
               'v1/finance/timeseries/' + symbol)
        with GuardedSession(impersonate='chrome') as session:
            response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        value = response.json()
        dump(path, value)
        dump(path.with_suffix('.source.json'), dict(url=url, params=params,
            retrieved_utc=datetime.now(timezone.utc).isoformat(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            historical_publication_vintage_verified=False))
    result = (value.get('timeseries') or {}).get('result') or []
    parts = [pd.DataFrame({'timestamp':x['timestamp'], 'shares':x['shares_out']})
             for x in result if 'shares_out' in x and 'timestamp' in x]
    if not parts:
        return dict(symbol=symbol, rows=0)
    df = pd.concat(parts, ignore_index=True)
    df['date'] = pd.to_datetime(df.timestamp, unit='s', utc=True).dt.strftime('%Y-%m-%d')
    df = df.sort_values('timestamp', kind='stable')
    n = df.groupby('date').shares.nunique()
    return dict(symbol=symbol, rows=len(df), first=df.date.min(), last=df.date.max(),
                conflicting_dates=int(n.gt(1).sum()),
                first_shares=float(df.shares.iloc[0]), last_shares=float(df.shares.iloc[-1]))


def summarize_cap(symbol, value):
    result = (value.get('timeseries') or {}).get('result') or []
    summary = {}
    for item in result:
        for kind in ['monthlyMarketCap', 'quarterlyMarketCap', 'trailingMarketCap']:
            rows = [r for r in item.get(kind, []) if r and r.get('asOfDate')]
            if rows:
                dates = sorted(r['asOfDate'] for r in rows)
                summary[kind] = dict(rows=len(rows), first=dates[0], last=dates[-1],
                    currencies=sorted(set(str(r.get('currencyCode')) for r in rows)),
                    earliest=rows[0], latest=rows[-1])
    return dict(symbol=symbol, series=summary, error=(value.get('timeseries') or {}).get('error'))


def screen():
    import yfinance as yf
    path = CACHE / ('us_screen_' + datetime.now(timezone.utc).date().isoformat() + '_listed_v1.json')
    if path.exists():
        result = json.loads(path.read_text())
    else:
        # Listed exchange groups; exclude OTC/grey market and ETFs. No domicile filter.
        q = yf.EquityQuery('and', [yf.EquityQuery('eq', ['region', 'us']),
            yf.EquityQuery('is-in', ['exchange', 'NMS', 'NGM', 'NCM', 'NYQ', 'ASE', 'PCX', 'BTS', 'CXI', 'NAE'])])
        result = yf.screen(q, size=250, offset=0,
                           sortField='intradaymarketcap', sortAsc=False,
                           session=GuardedSession(impersonate='chrome'))
        dump(path, result)
        dump(path.with_suffix('.source.json'), dict(
            query=q.to_dict(), size=250, offset=0, sortField='intradaymarketcap',
            retrieved_utc=datetime.now(timezone.utc).isoformat(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            limitation='Current candidate discovery only; never historical membership.'))
    quotes = result.get('quotes', [])
    print(json.dumps(dict(rows=len(quotes), total=result.get('total'),
        first40=[{k:r.get(k) for k in ['symbol','shortName','exchange','quoteType','marketCap','currency']}
                  for r in quotes[:40]]), indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', nargs='*', default=[])
    ap.add_argument('--screen', action='store_true')
    ap.add_argument('--shares', nargs='*', default=[])
    ap.add_argument('--cap-end', default=None)
    args = ap.parse_args()
    if args.screen:
        screen()
    summaries = []
    for symbol in args.symbols:
        summary = summarize_cap(symbol, fetch_cap(symbol, args.cap_end))
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    if summaries:
        dump(P / ('CAP_PROBE' + ('_before_' + args.cap_end if args.cap_end else '') + '.json'), summaries)
    summaries = []
    for symbol in args.shares:
        summary = fetch_shares(symbol)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    if summaries:
        dump(P / 'SHARES_PROBE.json', summaries)
