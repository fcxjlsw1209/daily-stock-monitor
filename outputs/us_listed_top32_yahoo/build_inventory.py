"""Build an independent current issuer Top32 and historical research inventory.

No current ranking is projected backward. Historical pool remains unavailable
until historical candidate coverage and cap units are checked.
"""
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

P = Path(__file__).resolve().parent
O = P.parent
CONFIG = json.loads((P / 'CONFIG.json').read_text())
CLASS_REPRESENTATIVES = {'GOOG': 'GOOGL', 'GOOGL': 'GOOGL',
                         'BRK-A': 'BRK-B', 'BRK-B': 'BRK-B',
                         'FOX': 'FOXA', 'FOXA': 'FOXA'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(name, value):
    (P / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def security_exclusion(quote):
    symbol = quote['symbol']
    if quote.get('exchange') not in CONFIG['us_exchange_codes']:
        return 'not_a_configured_US_listed_exchange'
    if quote.get('quoteType') != 'EQUITY':
        return 'not_equity'
    if quote.get('currency') != 'USD':
        return 'non_USD_needs_conversion'
    name = quote.get('longName', quote.get('shortName', '')).lower()
    if re.search(r'-P[A-Z]*$', symbol) or symbol == 'PBR-A' or 'preferred' in name:
        return 'preferred_equity_excluded'
    if re.search(r'\b(warrants?|rights?|units?)\b', name):
        return 'not_plain_common_equity_or_ADR'
    if not isinstance(quote.get('marketCap'), (int, float)) or quote['marketCap'] <= 0:
        return 'missing_positive_market_cap'
    return None


def current_pool(quotes):
    available = {q['symbol']: q for q in quotes if security_exclusion(q) is None}
    picked, dropped = [], []
    for q in quotes:
        s = q['symbol']
        reason = security_exclusion(q)
        representative = CLASS_REPRESENTATIVES.get(s, s)
        if not reason and representative != s:
            if representative not in available:
                raise ValueError('Missing predetermined common-class representative: ' + representative)
            reason = 'duplicate_issuer_use_' + representative
        if reason:
            dropped.append(dict(symbol=s, reason=reason))
            continue
        picked.append({k: q.get(k) for k in ['symbol', 'longName', 'exchange', 'quoteType',
                         'currency', 'marketCap', 'regularMarketTime', 'regularMarketPrice',
                         'firstTradeDateMilliseconds']})
    picked.sort(key=lambda r: (-r['marketCap'], r['symbol']))
    for n, r in enumerate(picked, 1):
        r['rank'] = n
        r['price_timestamp_utc'] = datetime.fromtimestamp(r['regularMarketTime'], timezone.utc).isoformat()
    if len(picked) < 32:
        raise ValueError('Too few issuer candidates')
    return picked, dropped


def cached_shares():
    rows = []
    for f in sorted((P / 'raw').glob('*_shares.json')):
        symbol = f.name.removesuffix('_shares.json')
        obj = json.loads(f.read_text())
        parts = [pd.DataFrame({'epoch':v['timestamp'], 'shares':v['shares_out']})
                 for v in (obj.get('timeseries') or {}).get('result', [])
                 if 'shares_out' in v and 'timestamp' in v]
        if not parts:
            continue
        df = pd.concat(parts)
        df['date'] = pd.to_datetime(df.epoch, unit='s', utc=True).dt.strftime('%Y-%m-%d')
        conflicts = df.groupby('date').shares.nunique().gt(1).sum()
        rows.append(dict(symbol=symbol, rows=len(df), first=df.date.min(), last=df.date.max(),
            conflicting_dates=int(conflicts), file=str(f), sha256=sha(f), source='new_Yahoo_response'))
    old = O / 'dynamic_top100/source_probes'
    for f in sorted(old.glob('*_yahoo_shares.csv')):
        symbol = f.name.removesuffix('_yahoo_shares.csv')
        if any(x['symbol'] == symbol for x in rows):
            continue
        df = pd.read_csv(f)
        dates = df.iloc[:, 0].astype(str).str.slice(0, 10)
        conflicts = pd.DataFrame({'date':dates, 'shares':df.iloc[:, 1]}).groupby('date').shares.nunique().gt(1).sum()
        rows.append(dict(symbol=symbol, rows=len(df), first=dates.min(), last=dates.max(),
            conflicting_dates=int(conflicts), file=str(f), sha256=sha(f), source='reused_Yahoo_cache'))
    return rows


def main():
    before = json.loads((O / 'monthly_universe_review/PROTECTED_BEFORE.json').read_text())
    assert all(Path(f).exists() and sha(f) == h for f, h in before.items())
    screen = sorted((P / 'raw').glob('us_screen_*_listed_v1.json'))[-1]
    raw = json.loads(screen.read_text())
    ranked, dropped = current_pool(raw['quotes'])
    # This list is explicitly a current snapshot. It never enters old backtests.
    snapshot = dict(version=CONFIG['version'],
        retrieved_utc=json.loads(screen.with_suffix('.source.json').read_text())['retrieved_utc'],
        stock32=ranked[:32], boundary33_to40=ranked[32:40],
        parent_screen_total=raw.get('total'), raw_candidates=len(raw['quotes']),
        filtered_issuer_candidates=len(ranked), skipped=dropped,
        share_class_rule='Predetermined GOOGL, BRK-B and FOXA representatives; do not sum duplicated company caps.',
        classification_status='Research snapshot; not exchange-certified listing master',
        historical_membership=False, source=str(screen), source_sha256=sha(screen))
    dump('CURRENT_TOP32.json', snapshot)
    dump('CURRENT_RANKED_CANDIDATES.json', ranked)
    shares = cached_shares()
    dump('SHARES_COVERAGE.json', shares)

    listing_dir = O / 'dynamic_top100/additional_sources/alpha_vantage'
    listing_files = sorted(listing_dir.glob('*_active.csv')) + sorted(listing_dir.glob('*_delisted.csv'))
    records = []
    for f in listing_files:
        frame = pd.read_csv(f, keep_default_na=False)
        frame = frame[frame.assetType.eq('Stock')].copy()
        frame['source_file'] = f.name
        records.extend(frame.to_dict('records'))
    # Retain distinct listing episodes and names; symbol alone is not identity.
    keys = ['symbol', 'name', 'exchange', 'ipoDate', 'delistingDate', 'status']
    by_episode = {}
    for r in records:
        k = tuple(r[x] for x in keys)
        if k not in by_episode:
            by_episode[k] = dict(zip(keys, k), evidence_files=[])
        by_episode[k]['evidence_files'].append(r['source_file'])
    episodes = list(by_episode.values())
    dump('LISTING_EVIDENCE.json', episodes)

    cap_probe = json.loads((P / 'CAP_PROBE.json').read_text())
    old_pools = json.loads((O / 'monthly_universe_review/MONTHLY_POOLS.json').read_text())
    symbols = set(s for r in old_pools for s in r['stock32'])
    # Discovery only: all raw current common candidates, old price/cap archives and
    # the full listing evidence stay separate from historical eligibility.
    stock_manifest = dict(
        preferred_current_classes=CLASS_REPRESENTATIVES,
        current_discovery_symbols=[r['symbol'] for r in ranked],
        previous_proxy_ever_top32=sorted(symbols),
        listing_evidence_records=len(episodes),
        listing_distinct_symbols=len({r['symbol'] for r in episodes}),
        listing_source_files=[dict(file=str(f), sha256=sha(f)) for f in listing_files],
        candidate_coverage_rule='Historical scope is all US-listed common-equity issuers; current discovery candidates are not a historical screen.',
        historical_ranked_membership_file=None)
    dump('UNIVERSE_MANIFEST.json', stock_manifest)
    summary = dict(
        scope=CONFIG['requested_scope'], current_snapshot_ready=True,
        current_stock_count=32, current_parent_screen_total=raw.get('total'),
        successful_historical_shares=len(shares),
        new_share_probe_symbols=[x['symbol'] for x in shares if x['source']=='new_Yahoo_response'],
        reused_share_probe_symbols=[x['symbol'] for x in shares if x['source']=='reused_Yahoo_cache'],
        direct_monthly_cap_probe=[dict(symbol=x['symbol'], **{k:v for k,v in x['series']['monthlyMarketCap'].items() if k in ['rows','first','last']}) for x in cap_probe],
        historical_ready=False, historical_backtest_run=False, model_search_paused=True,
        remaining=['Recover broader historical large-cap candidate coverage from listing evidence and cap archives, without restricting to present survivors or indices.',
                   'Align past Yahoo shares with actual split and ADR units; reconcile multiple conflicting values on the same source date.',
                   'Build daily/month-end Top32, audit boundary ranks and price warmup; never backfill current ranking.',
                   'Keep the 17-ETF list explicitly hypothetical until independent evidence identifies the full author list.'],
        protected_files_unchanged=len(before), observer_changed=False,
        comparison_cost_bps=0, updated_utc=datetime.now(timezone.utc).isoformat())
    dump('STATUS.json', summary)
    hashes = {str(screen):sha(screen), str(P/'CONFIG.json'):sha(P/'CONFIG.json')}
    hashes.update({x['file']:x['sha256'] for x in shares})
    hashes.update({str(f):sha(f) for f in listing_files})
    dump('SOURCE_HASHES.json', hashes)
    assert len({r['symbol'] for r in ranked[:32]}) == 32
    assert all(r['currency']=='USD' for r in ranked[:32])
    assert all(sha(f)==h for f,h in before.items())
    dump('VALIDATION.json', dict(protected_files_unchanged=len(before), stock32_count=32,
          no_duplicate_known_issuers=True, no_current_to_historical_backfill=True,
          sources_checked=len(hashes), price_dates=sorted({r['price_timestamp_utc'] for r in ranked[:32]})))
    lines = ['All-US-listed issuer Top32: Yahoo feasibility and universe inventory', '',
        'Current snapshot: 32 issuers, separate from historical backtest membership.',
        'Historical Top32 reconstruction is not yet complete. No fitting or paper ledger changes.',
        'Comparison transaction costs: zero.', '',
        'Current Top32 (Yahoo regular-session quote dates recorded per row):']
    lines += [f"{r['rank']:2d}. {r['symbol']:6s} USD {r['marketCap']/1e9:.2f}bn" for r in ranked[:32]]
    lines += ['', 'Historical Yahoo shares coverage:']
    lines += [f"{r['symbol']}: {r['first']} to {r['last']}, {r['rows']} rows; {r['conflicting_dates']} conflicting dates, pending reconciliation." for r in shares]
    lines += ['', 'Direct market-cap probes: six symbols each returned 15 monthly points, 2025-06-30 to 2026-08-31.',
        'COST/BABA requests ending 2023-01-01 returned no direct cap points; no free full-period direct ranking verified.',
        '', 'Remaining work:'] + summary['remaining']
    (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    # User-authorized research routing only. No existing loader or observer changes.
    pointer = O/'monthly_universe_review/ACTIVE_RESEARCH_UNIVERSE.json'
    pointer.write_text(json.dumps(dict(version=CONFIG['version'], config=str(P/'CONFIG.json'),
        status=str(P/'STATUS.json'), historical_ready=False,
        allow_legacy_proxy_fallback=False, fitting_paused=True), indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
