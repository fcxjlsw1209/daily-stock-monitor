"""Standalone retrospective monthend signal. Never writes an observation ledger."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import requests

P = Path(__file__).resolve().parent
O = P.parent
DAY = '2026-09-30'
sys.path.insert(0, str(O / 'us_listed_top32_yahoo'))
from collect_monthly_rankings import parse_ranking, filter_us
sys.path.insert(0, str(O / 'monthly_guidance_reset'))
from study import load_prices, actual, display
sys.path.insert(0, str(O / 'monthly_regime_monitor'))
from strategy import selection


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(name, x):
    (P / name).write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def ranks():
    path = P / 'market_cap_2026-09-30.html'
    if not path.exists():
        url = 'https://companiesmarketcap.com/time-machine/' + DAY + '/'
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
        response.raise_for_status()
        parse_ranking(response.content, DAY)
        path.write_bytes(response.content)
        dump('RANK_SOURCE.json', dict(url=url, response_url=response.url, sha256=sha(path), retrieved_utc=datetime.now(timezone.utc).isoformat()))
    assert sha(path) == read(P / 'RANK_SOURCE.json')['sha256']
    rows = parse_ranking(path.read_bytes(), DAY)
    evidence = read(O / 'us_listed_top32_yahoo/LISTING_EVIDENCE.json')
    catalog = read(O / 'us_listed_top32_yahoo/broad_caps/CATALOG.json')
    chosen = filter_us(rows, DAY, evidence, catalog)
    dump('POOL.json', dict(date=DAY, **chosen, limit='Retrospective vendor ranking, filtered using cached US-listing evidence; not original vintage certified.'))
    print('TOP32', [r['symbol'] for r in chosen['top32']], flush=True)


def panel():
    adj, raw, members, periods, sources, pools = load_prices()
    config = read(O / 'monthly_regime_monitor/config.json')
    stocks = [actual(r['symbol']) for r in read(P / 'POOL.json')['top32']]
    legacy = pd.read_csv(O / 'monthly_reconstruction_v2/foreign/membership.csv.gz', index_col=0, parse_dates=True)
    legacy_stocks = legacy.loc[DAY].index[legacy.loc[DAY].fillna(False).astype(bool)].tolist()
    required = sorted(set(stocks + legacy_stocks + config['etfs']))
    frames = dict(adj=adj, **{('open' if k == 'opening' else k): v for k, v in raw.items() if k != 'sector60'})
    for k in frames:
        frames[k] = frames[k].reindex(columns=sorted(set(frames[k].columns) | set(required))).copy()
    for f in (P / 'prices').glob('*.csv'):
        md = read(f.with_suffix('.source.json'))
        assert sha(f) == md['sha256']
        z = pd.read_csv(f, index_col=0, parse_dates=True)
        for k, col in [('adj', 'Adj Close'), ('open', 'Open'), ('close', 'Close'), ('high', 'High'), ('low', 'Low'), ('volume', 'Volume')]:
            frames[k][actual(md['symbol'])] = z[col].reindex(adj.index)
        sources[str(f)] = sha(f)
    return frames, required, stocks, legacy_stocks, config, sources


def missing(frames, required):
    needed = []
    for s in required:
        tail = frames['adj'].loc[:DAY, s].tail(400)
        # Include ample context for volatility-regime threshold and warmup.
        enough = tail.notna().sum() >= 295 and tail.tail(295).notna().all()
        factors = all(frames[k].loc[:DAY, s].tail(126).notna().all() for k in ['close', 'high', 'low', 'volume'])
        fill = all(pd.notna(frames[k].at[DAY, s]) for k in ['close', 'adj']) and pd.notna(frames['open'].at['2026-10-01', s])
        if not (enough and factors and fill):
            needed.append(s)
    return needed


def prices():
    import yfinance as yf
    frames, required, stocks, legacy, config, sources = panel()
    todo = missing(frames, required)
    print('PRICE_GAPS', todo, flush=True)
    dest = P / 'prices'
    dest.mkdir(exist_ok=True)
    rate = P / 'YAHOO_RATE_LIMIT.json'
    if rate.exists() and read(rate)['date'] == datetime.now(timezone.utc).date().isoformat():
        raise RuntimeError('Yahoo already rate limited today; no retry')
    failures = []
    for s in todo:
        symbol = display(s)
        try:
            z = yf.Ticker(symbol).history(start='2024-01-01', end='2026-10-02', auto_adjust=False, actions=True, raise_errors=True)
            if z.empty or 'Adj Close' not in z:
                raise ValueError('Empty/incomplete Yahoo history')
            z.index = z.index.tz_localize(None).normalize()
            assert z.index.is_unique and z.index.is_monotonic_increasing
            assert DAY in z.index.astype(str)
            assert (z.Close.dropna() > 0).all() and (z['Adj Close'].dropna() > 0).all()
            f = dest / (symbol + '.csv')
            z.to_csv(f)
            f.with_suffix('.source.json').write_text(json.dumps(dict(symbol=symbol, source='Yahoo via yfinance',
                requested_start='2024-01-01', end_exclusive='2026-10-02', retrieved_utc=datetime.now(timezone.utc).isoformat(),
                sha256=sha(f), rows=len(z), first=str(z.index[0].date()), last=str(z.index[-1].date())), indent=2) + '\n')
            print('SAVED', symbol, len(z), flush=True)
        except Exception as exc:
            if '429' in str(exc) or 'RateLimit' in type(exc).__name__:
                dump('YAHOO_RATE_LIMIT.json', dict(date=datetime.now(timezone.utc).date().isoformat(), symbol=symbol))
                raise
            failures.append(dict(symbol=symbol, error=str(exc)[:250]))
            print('FAILED', symbol, str(exc)[:160], flush=True)
    dump('PRICE_FAILURES.json', failures)


def calculate():
    frames, required, stocks, legacy, config, sources = panel()
    todo = missing(frames, required)
    # IPOs without 295 prior sessions remain valid members but are ineligible;
    # do not drop them from the cross-sectional factor pool.
    missing_current = [s for s in todo if pd.isna(frames['adj'].at[DAY, s]) or pd.isna(frames['close'].at[DAY, s])]
    assert not missing_current, ('Unresolved signal-day prices', missing_current)
    results = {}
    for name, ss in [('published_Top32', stocks), ('registered_proxy_Top32', legacy)]:
        member = pd.DataFrame(False, index=frames['adj'].index, columns=frames['adj'].columns)
        member.loc[DAY, ss] = True
        p = dict(frames, member=member)
        chosen = selection(p, DAY, config['etfs'])
        rows = []
        for s in chosen['symbols']:
            assert s not in todo, ('Insufficient selected history', s)
            rows.append(dict(ticker=display(s), score=chosen['scores'][s], weight=chosen['target_weights'][s],
                             signal_close_raw=float(frames['close'].at[DAY, s]),
                             signal_close_adjusted=float(frames['adj'].at[DAY, s]),
                             next_open_raw=float(frames['open'].at['2026-10-01', s])))
        results[name] = dict(selected=rows, universe=[display(s) for s in chosen['universe']], eligible_count=chosen['candidate_count'])
    dump('RESULTS.json', dict(signal_date=DAY, hypothetical_month='2026-10', execution_reference='Same September30 close requested; October1 raw open separately for registered next-open convention.',
        purpose='User-authorized retrospective calculation, no paper ledger backfill, no actual orders.', strategy_weights=config['score_weights'],
        results=results, incomplete_history_members=[display(s) for s in todo],
        source_limit='Published ranking and registered estimated-cap proxy shown separately. No current-rank substitution or author-label input.',
        source_hashes=sources))
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['ranks', 'prices', 'calculate'])
    globals()[parser.parse_args().phase]()
