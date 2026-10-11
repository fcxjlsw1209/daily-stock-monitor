"""Validate research price caches without promoting ticker matches to identities.

Output remains separate from all paper ledgers. Suspicious identities are retained
in quarantine; no missing observations or corporate actions are manufactured.
"""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from research_calendar import sessions as research_sessions

P = Path(__file__).resolve().parent
SUSPECT = {
    'ACT': 'Current Enact history does not establish historical Actavis equity.',
    'MICC': 'Current Magnum Ice Cream history does not establish Millicom equity.',
    'SLE': 'Current ticker history does not establish historical Sara Lee equity.',
    'GOLD': 'Reused ticker: historical Randgold security requires explicit mapping.',
    'NIHD': 'Post-bankruptcy equity must not be spliced to pre-bankruptcy NII.',
    'AA': 'Alcoa separation requires security and adjustment reconciliation.',
}


def validate_frame(d):
    """Return row flags; valid bars alone do not prove security identity."""
    prices = d[['open', 'high', 'low', 'close', 'adj_close']].apply(pd.to_numeric, errors='coerce')
    vol = pd.to_numeric(d.volume, errors='coerce')
    valid = np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
    tol = prices[['open', 'high', 'low', 'close']].max(axis=1) * 1e-6
    valid &= (prices.high + tol >= prices[['open', 'close', 'low']].max(axis=1))
    valid &= (prices.low - tol <= prices[['open', 'close']].min(axis=1))
    valid &= np.isfinite(vol) & (vol >= 0)
    dates = pd.to_datetime(d.date, errors='coerce', utc=True)
    valid &= dates.notna() & ~dates.duplicated(keep=False)
    return valid



def session_mask(dates):
    parsed = pd.to_datetime(dates, errors='coerce')
    if parsed.notna().sum() == 0:
        return pd.Series(False, index=dates.index)
    calendar = research_sessions(parsed.min(), parsed.max())
    return dates.isin(set(calendar.strftime('%Y-%m-%d')))


def read_quantquote(f):
    d = pd.read_csv(f, header=None)
    if len(d.columns) != 7:
        raise ValueError('QuantQuote archive requires exactly seven columns')
    d.columns = ['date','time','open','high','low','close','volume']
    if not pd.to_numeric(d.time, errors='coerce').eq(0).all():
        raise ValueError('Expected daily QuantQuote bars, not intraday')
    d['date'] = pd.to_datetime(d.date.astype(str), format='%Y%m%d', errors='raise').dt.strftime('%Y-%m-%d')
    d['adj_close'] = d.close
    return d


def read_cache(f, provider):
    prov = json.loads(f.with_suffix('.provenance.json').read_text())
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    if sha != prov['sha256']:
        raise ValueError('Cache checksum mismatch: ' + str(f))
    if provider == 'quantiacs_archive':
        d = pd.read_csv(f)
        symbol = f.stem
        basis = 'quantiacs_split_adjusted_candidate_cash_dividend_total_return_pending'
    elif provider == 'union_intraday_archive':
        d = pd.read_csv(f)
        symbol = f.stem
        basis = 'vendor_adjusted_regular_session_intraday_proxy'
    elif provider in ['pystock_initial_archive', 'pystock_daily_archive']:
        d = pd.read_csv(f)
        symbol = f.stem
        basis = 'archived_yahoo_snapshot_adjustment_reconciliation_required'
    elif provider == 'wiki_union_archive':
        d = pd.read_csv(f)
        symbol = f.stem
        basis = 'quandl_wiki_separate_raw_and_adjusted_ohlcv'
    elif provider == 'quantquote_archive':
        d = read_quantquote(f)  # Source basis remains explicitly uncertified.
        symbol = f.stem
        basis = 'quantquote_archived_adjusted_ohlcv_reconciliation_required'
    elif provider == 'teddy_archive':
        d = pd.read_csv(f)
        d['adj_close'] = d['close']  # Source generator explicitly outputs adjusted OHLC.
        symbol = f.stem
        basis = 'archived_quandl_or_yahoo_adjusted_ohlcv_raw_units_unverified'
    elif provider in ['yahoo', 'qstk_archive', 'kaggle2020_archive']:
        d = pd.read_csv(f).rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume',
            'Dividends': 'dividend', 'Stock Splits': 'split'})
        symbol = f.stem
        # Yahoo historical OHLC/volume are split adjusted even with auto_adjust=False.
        basis = ('yahoo_split_adjusted_ohlcv_dividend_adjusted_close' if provider=='yahoo'
                 else 'archived_yahoo_ohlcv_adjustment_basis_requires_reconciliation')
    else:
        raw = json.loads(f.read_text())
        if not isinstance(raw, list) or not raw:
            return None
        d = pd.DataFrame(raw).rename(columns={'adjClose': 'adj_close', 'divCash': 'dividend',
                                              'splitFactor': 'split'})
        symbol = f.name.removesuffix('_prices.json')
        basis = 'tiingo_raw_ohlcv_and_separate_adjusted_close'
    d['date'] = pd.to_datetime(d.date, utc=True).dt.strftime('%Y-%m-%d')
    valid = validate_frame(d)
    valid &= session_mask(d.date)
    d['symbol'] = symbol
    d['provider'] = provider
    d['price_basis'] = basis
    d['source_sha256'] = sha
    d['bar_valid'] = valid
    d['positive_trading_volume'] = pd.to_numeric(d.volume, errors='coerce').gt(0)
    d['identity_verified'] = False
    d['quarantine_reason'] = np.where(valid, '', 'invalid_or_duplicate_bar')
    if symbol == 'LIFE' and provider == 'pystock_daily_archive':
        d['quarantine_reason'] = 'Reused LIFE ticker: 2015 aTyr is not historical Life Technologies.'
    historical_act = symbol == 'ACT' and provider in ['pystock_initial_archive','pystock_daily_archive']
    if symbol in SUSPECT and not (historical_act or (provider == 'qstk_archive' and symbol in ['AA','SLE']) or (provider == 'wiki_union_archive' and symbol == 'GOLD')):
        d['quarantine_reason'] = SUSPECT[symbol]
    for event in json.loads((P/'REVIEWED_TERMINAL_EVENTS.json').read_text()):
        if event['symbol'] == symbol:
            d.loc[d.date.ge(event['quarantine_from']), 'quarantine_reason'] = 'on_or_after_completed_acquisition'
    return d, dict(symbol=symbol, provider=provider, rows=len(d), valid_bars=int(valid.sum()),
                  quarantined_rows=int(d.quarantine_reason.ne('').sum()), first=d.date.min(),
                  last=d.date.max(), source=str(f), sha256=sha, identity_verified=False)


def main():
    out = P / 'normalized_prices'
    out.mkdir(exist_ok=True)
    inventory = []
    files = [(f, 'yahoo') for f in sorted((P / 'sources/yahoo_prices').glob('*.csv'))]
    files += [(P / 'sources/teddy' / (s+'.csv'), 'teddy_archive') for s in ['DELL','SNDK','KRFT','CERN','CELG','XLNX','FOXA','FOX','AABA']]
    files += [(f, 'quantiacs_archive') for f in sorted((P / 'sources/quantiacs/extracted').glob('*.csv'))]
    files += [(f, 'kaggle2020_archive') for f in sorted((P / 'sources/kaggle2020').glob('*.csv'))]
    files += [(f, 'union_intraday_archive') for f in sorted((P / 'sources/union_intraday').glob('*.csv'))]
    files += [(f, 'pystock_initial_archive') for f in sorted((P / 'sources/pystock/extracted').glob('*.csv'))]
    files += [(f, 'pystock_daily_archive') for f in sorted((P / 'sources/pystock/daily_extracted').glob('*.csv'))]
    files += [(f, 'wiki_union_archive') for f in sorted((P / 'sources/wiki_union').glob('*.csv'))]
    files += [(f, 'quantquote_archive') for f in sorted((P / 'sources/quantquote').glob('*.csv'))]
    files += [(f, 'qstk_archive') for f in sorted((P / 'sources/qstk').glob('*.csv'))]
    symbols = 'CTRX FWLT MICC NIHD NUAN SGEN SHPG STRZA VMED WCRX SPLK JOY BB QVCA ANSS WBA EA CCEP TTD MTCH XLNX CERN CELG SYMC'.split()
    files += [(P.parent / 'dynamic_top100/additional_sources/tiingo' / (s+'_prices.json'), 'tiingo') for s in symbols]
    for f, provider in files:
        result = read_cache(f, provider)
        if result is None:
            continue
        d, info = result
        dest = out / (info['symbol']+'_'+provider+'.csv.gz')
        d.to_csv(dest, index=False, compression={'method': 'gzip', 'mtime': 0})
        info['normalized_file'] = str(dest.relative_to(P))
        info['normalized_sha256'] = hashlib.sha256(dest.read_bytes()).hexdigest()
        inventory.append(info)
    report = dict(checked_at=datetime.now(timezone.utc).isoformat(), files=inventory,
                  rows=sum(x['rows'] for x in inventory),
                  quarantined_rows=sum(x['quarantined_rows'] for x in inventory),
                  identity_certified=False, admission=False)
    (P/'NEW_PRICE_VALIDATION.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
