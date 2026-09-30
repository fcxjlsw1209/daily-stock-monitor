"""Explicit YXL-inspired proxy. Never imports or executes the paper monitors."""
from pathlib import Path
from dataclasses import dataclass, asdict, replace
import os
import hashlib
import json
import sqlite3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('STOCK_RESEARCH_OUTPUTS', str(ROOT.parents[1] / 'outputs'))).expanduser().resolve()


@dataclass(frozen=True)
class Spec:
    name: str
    inner: str = 'EMA'
    fast: int = 10
    slow: int = 16
    outer: str = 'EMA'
    smooth: int = 3
    gap: int = 3
    secondary: str = 'daily'
    trigger: str = 'combined'
    exit_mode: str = 'main'


def specifications():
    result = []
    for name, inner, fast, slow, outer, smooth, gap in [
        ('default', 'EMA', 10, 16, 'EMA', 3, 3),
        ('hma58', 'HMA', 5, 8, 'RMA', 8, 4),
        ('hma612', 'HMA', 6, 12, 'RMA', 8, 4),
    ]:
        for sec in ['none', 'daily', 'weekly']:
            result.append(Spec(f'{name}_{sec}', inner, fast, slow, outer, smooth, gap, sec))
    base = next(x for x in result if x.name == 'default_daily')
    result += [replace(base, name='default_cross', trigger='cross'),
               replace(base, name='default_basis', trigger='basis'),
               replace(base, name='default_exit_either', exit_mode='either'),
               replace(base, name='default_exit_both', exit_mode='both'),
               replace(base, name='default_state', trigger='state')]
    return result


def rma(s, n):
    # Wilder seed: SMA of first n finite inputs, then alpha=1/n.
    a = s.to_numpy(float)
    out = np.full(len(a), np.nan)
    finite = np.flatnonzero(np.isfinite(a))
    if len(finite) < n:
        return pd.Series(out, index=s.index)
    start = finite[n - 1]
    if not np.isfinite(a[finite[0]:]).all():
        raise ValueError('Interior missing inputs to RMA')
    out[start] = a[finite[:n]].mean()
    for i in range(start + 1, len(a)):
        out[i] = (out[i - 1] * (n - 1) + a[i]) / n
    return pd.Series(out, index=s.index)


def ma(s, n, kind):
    if kind == 'EMA':
        return s.ewm(span=n, adjust=False).mean()
    if kind == 'RMA':
        return rma(s, n)
    if kind == 'WMA':
        w = np.arange(1, n + 1, dtype=float)
        return s.rolling(n).apply(lambda a: np.dot(a, w) / w.sum(), raw=True)
    if kind == 'HMA':
        return ma(2 * ma(s, max(1, n // 2), 'WMA') - ma(s, n, 'WMA'),
                  max(1, int(np.sqrt(n) + .5)), 'WMA')
    raise ValueError(kind)


def rsi(c, n=14):
    change = c.diff()
    up, down = rma(change.clip(lower=0), n), rma(-change.clip(upper=0), n)
    ans = 100 - 100 / (1 + up / down)
    return ans.mask((up == 0) & (down == 0), 50).mask((down == 0) & (up > 0), 100)


def prior_week_rsi(c):
    # Entire prior W-FRI week only. Current week's final close never leaks backward.
    week = c.index.to_period('W-FRI')
    weekly = c.groupby(week).last()
    values = rsi(weekly).shift(1)
    return pd.Series(values.reindex(week).to_numpy(), index=c.index)


def cross(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))


def features(d, spec):
    src = (d.High + d.Low) / 2
    fast = ma(ma(src, spec.fast, spec.inner), spec.smooth, spec.outer)
    slow = ma(ma(src, spec.slow, spec.inner), spec.smooth, spec.outer)
    basis = (fast + slow) / 2
    tr = pd.concat([d.High - d.Low, (d.High - d.Close.shift()).abs(),
                    (d.Low - d.Close.shift()).abs()], axis=1).max(axis=1)
    atr = rma(tr, 14)
    macd = ma(d.Close, 12, 'EMA') - ma(d.Close, 26, 'EMA')
    hist = macd - ma(macd, 9, 'EMA')
    rs = rsi(d.Close)
    weekly = prior_week_rsi(d.Close)
    bull, bear = (hist > 0) & (rs > 50), (hist < 0) & (rs < 50)
    buy, sell = cross(fast, slow), cross(slow, fast)
    if spec.trigger == 'basis':
        buy, sell = cross(d.Close, basis), cross(basis, d.Close)
    elif spec.trigger == 'combined':
        buy = buy | cross(d.Close, basis) | ((fast > slow) & (d.Low <= fast) &
              (d.Close > fast) & (d.Close.shift() > fast.shift()))
        sell = sell | cross(basis, d.Close) | ((fast < slow) & (d.High >= fast) &
               (d.Close < fast) & (d.Close.shift() < fast.shift()))
    elif spec.trigger == 'state':
        buy, sell = (fast > slow) & (d.Close > basis), (fast < slow) | (d.Close < basis)
    raw_buy, raw_sell = buy.to_numpy().copy(), sell.to_numpy().copy()
    if spec.trigger != 'state':
        last = -100000
        for i in range(len(d)):
            if raw_buy[i] == raw_sell[i] or i - last < spec.gap:
                buy.iloc[i] = sell.iloc[i] = False
            elif raw_buy[i] or raw_sell[i]:
                last = i
    if spec.secondary != 'none':
        buy &= bull
    if spec.secondary == 'weekly':
        buy &= weekly > 50
    if spec.exit_mode == 'either':
        sell |= bear
    elif spec.exit_mode == 'both':
        sell &= bear
    # A common warmup across variants. Used only after enough actual daily bars.
    buy.iloc[:200] = False
    sell.iloc[:200] = False
    return pd.DataFrame(dict(fast=fast, slow=slow, basis=basis, atr=atr,
                             macd_hist=hist, rsi=rs, prior_week_rsi=weekly,
                             buy=buy, sell=sell), index=d.index)


def validate(d):
    assert d.index.is_monotonic_increasing and not d.index.has_duplicates
    assert np.isfinite(d[['Open', 'High', 'Low', 'Close', 'Dividends']]).all().all()
    assert (d[['Open', 'High', 'Low', 'Close']] > 0).all().all()
    assert (d.High + 1e-6 >= d[['Open', 'Close', 'Low']].max(axis=1)).all()
    assert (d.Low - 1e-6 <= d[['Open', 'Close', 'High']].min(axis=1)).all()
    assert (d.Dividends >= 0).all()


def simulate(d, f, start, fee_bps=5, hold=False):
    cash, qty, initial = 100000., 0., 100000.
    half = fee_bps / 10000
    dates = d.index
    daily, trades, orders = [], [], []
    cost, trade_div, fees, dividend_total = 0., 0., 0., 0.
    entry_i, entry_date, entry_price, entry_fee = None, None, None, None
    ever_bought = False
    opening, close, div = (d[k].to_numpy(float) for k in ['Open', 'Close', 'Dividends'])
    buy, sell = f.buy.to_numpy(bool), f.sell.to_numpy(bool)
    for i, day in enumerate(dates):
        if day < pd.Timestamp(start):
            continue
        if i == 0:
            raise ValueError('Needs previous session')
        receipt = qty * div[i]
        cash += receipt
        trade_div += receipt
        dividend_total += receipt
        sold = False
        if qty and sell[i - 1] and not hold:
            proceeds = qty * opening[i]
            fee = proceeds * half
            cash += proceeds - fee
            fees += fee
            pnl = proceeds - fee + trade_div - cost
            trades.append(dict(entry_date=str(entry_date.date()), exit_date=str(day.date()),
                               entry_price=entry_price, exit_price=opening[i], qty=qty,
                               entry_outlay=cost, dividends=trade_div, entry_fee=entry_fee,
                               exit_fee=fee, pnl=pnl, return_net=pnl/cost, sessions=i-entry_i))
            orders.append(dict(date=str(day.date()), signal_date=str(dates[i-1].date()),
                               side='sell', price=opening[i], qty=qty, fee=fee))
            qty, cost, trade_div = 0., 0., 0.
            sold = True
        if not qty and not sold and ((hold and not ever_bought) or (not hold and buy[i-1] and not sell[i-1])):
            cost = cash
            qty = cash / (opening[i] * (1 + half))
            entry_fee = qty * opening[i] * half
            fees += entry_fee
            cash = 0.
            entry_i, entry_date, entry_price = i, day, opening[i]
            ever_bought = True
            orders.append(dict(date=str(day.date()), signal_date=str(dates[i-1].date()),
                               side='buy', price=opening[i], qty=qty, fee=entry_fee))
        eq = cash + qty * close[i]
        assert eq > 0 and cash >= -1e-6
        daily.append(dict(date=day, equity=eq, cash=cash, exposure=qty*close[i]/eq,
                          fees=fees, dividends=dividend_total))
    out = pd.DataFrame(daily).set_index('date')
    out['return'] = out.equity.pct_change().fillna(out.equity.iloc[0]/initial-1)
    open_pnl = qty * close[-1] + trade_div - cost if qty else 0.
    assert abs(out.equity.iloc[-1] - initial - sum(x['pnl'] for x in trades) - open_pnl) < initial * 1e-8
    return out, trades, orders, dict(open_position=bool(qty), open_pnl=open_pnl,
                                    final_equity=out.equity.iloc[-1], fees=fees)


def summarize(daily, trades, start, end):
    d = daily.loc[start:end]
    if d.empty:
        return {}
    net = (1+d['return']).cumprod()
    peak = np.maximum.accumulate(np.r_[1., net.to_numpy()])[1:]
    years = ((d.index[-1]-d.index[0]).days+1)/365.25
    t = [x for x in trades if start <= x['exit_date'] <= end]
    wins = [x['pnl'] for x in t if x['pnl'] > 0]
    losses = [x['pnl'] for x in t if x['pnl'] < 0]
    return dict(start=str(d.index[0].date()), end=str(d.index[-1].date()), days=len(d),
                cagr=float(net.iloc[-1]**(1/years)-1), total_return=float(net.iloc[-1]-1),
                max_drawdown=float((net/peak-1).min()), closed_trades=len(t),
                win_rate=len(wins)/len(t) if t else None,
                profit_factor=sum(wins)/-sum(losses) if losses else None,
                mean_trade_return=float(np.mean([x['return_net'] for x in t])) if t else None,
                mean_hold=float(np.mean([x['sessions'] for x in t])) if t else None,
                exposure=float(d.exposure.mean()),
                sharpe_zero_rf=float(d['return'].mean()/d['return'].std()*np.sqrt(252))
                if d['return'].std() > 0 else None)


def read_csv(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)


def main():
    dest = ROOT/'results'
    dest.mkdir(exist_ok=True)
    files = [SOURCE/'qqq_ma200_research/data/QQQ.csv',
             SOURCE/'overnight_research/spy_source.csv', SOURCE/'stock_database/stocks.sqlite']
    manifest = {str(p): dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size) for p in files}
    (dest/'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    specs = specifications()
    (dest/'specifications.json').write_text(json.dumps([asdict(x) for x in specs], indent=2))
    etfs = {'QQQ': read_csv(files[0]), 'SPY': read_csv(files[1])}
    metrics, years, daily_frames, trade_rows, order_rows, open_rows = [], [], [], [], [], []
    for symbol, data in etfs.items():
        validate(data)
        start = '2000-01-01' if symbol == 'QQQ' else '2011-01-01'
        end = str(data.index[-1].date())
        for spec in specs + [Spec('buy_hold')]:
            f = features(data, spec)
            if spec.name in ['default_daily', 'hma58_daily', 'hma612_daily']:
                f.to_csv(dest/f'{symbol}_{spec.name}_signals.csv.gz', compression='gzip')
            for fee in [5, 10]:
                d, t, o, u = simulate(data, f, start, fee, hold=spec.name=='buy_hold')
                tags = dict(symbol=symbol, model=spec.name, fee_bps_per_side=fee)
                for label, a, b in [('all', start, end), ('since2011','2011-01-01',end),
                                     ('since2019','2019-01-01',end), ('since2022','2022-01-01',end),
                                     ('early','2011-01-01','2018-12-31')]:
                    metrics.append(dict(**tags, window=label, **summarize(d,t,a,b)))
                if fee == 5:
                    daily_frames.append(d.assign(**tags).reset_index())
                    trade_rows.extend([dict(**tags,**row) for row in t])
                    order_rows.extend([dict(**tags,**row) for row in o])
                    open_rows.append(dict(**tags, **u))
                    for y in range(int(start[:4]), data.index[-1].year+1):
                        years.append(dict(**tags, year=y, **summarize(d,t,f'{y}-01-01',f'{y}-12-31')))
        print(symbol, 'completed', len(data), 'input rows', flush=True)
    pd.DataFrame(metrics).to_csv(dest/'etf_metrics.csv',index=False)
    pd.DataFrame(years).to_csv(dest/'etf_yearly.csv',index=False)
    pd.concat(daily_frames).to_csv(dest/'etf_equity.csv.gz',index=False,compression='gzip')
    pd.DataFrame(trade_rows).to_csv(dest/'etf_trades.csv',index=False)
    pd.DataFrame(order_rows).to_csv(dest/'etf_orders.csv',index=False)
    (dest/'etf_open_positions.json').write_text(json.dumps(open_rows,indent=2))

    with sqlite3.connect(f'file:{files[2]}?mode=ro', uri=True) as db:
        prices = pd.read_sql_query('SELECT symbol,trade_date,open,high,low,close,dividends FROM daily_prices ORDER BY trade_date',db)
    stock_metrics, exclusions, stock_daily, stock_trades, stock_orders = [], [], [], [], []
    cols = {'trade_date':'Date','open':'Open','high':'High','low':'Low','close':'Close','dividends':'Dividends'}
    for symbol, frame in prices.groupby('symbol'):
        d = frame.rename(columns=cols).drop(columns='symbol')
        d.Date = pd.to_datetime(d.Date)
        d = d.set_index('Date')
        if symbol == 'SHEL':
            exclusions.append(dict(symbol=symbol,reason='Known OHLC inconsistency 2012-05-23; exclude entire series'))
            continue
        if (d.index < pd.Timestamp('2019-01-01')).sum() < 200:
            exclusions.append(dict(symbol=symbol,reason='Fewer than 200 pre-2019 sessions'))
            continue
        validate(d)
        ref_dates = etfs['SPY'].loc['2019-01-01':'2026-09-18'].index
        if not d.loc['2019-01-01':].index.equals(ref_dates):
            raise ValueError('Missing sessions '+symbol)
        for spec in specs + [Spec('buy_hold')]:
            f = features(d,spec)
            eq,t,o,u = simulate(d,f,'2019-01-01',5,hold=spec.name=='buy_hold')
            tags = dict(symbol=symbol,model=spec.name)
            stock_metrics.append(dict(**tags,**summarize(eq,t,'2019-01-01','2026-09-18'),**u))
            if spec.name in ['default_daily','default_none','hma58_daily','hma612_daily','buy_hold']:
                stock_daily.append(eq.assign(**tags).reset_index())
                stock_trades.extend([dict(**tags,**row) for row in t])
                stock_orders.extend([dict(**tags,**row) for row in o])
        print(symbol,'complete',flush=True)
    pd.DataFrame(stock_metrics).to_csv(dest/'stock_metrics.csv',index=False)
    pd.DataFrame(exclusions).to_csv(dest/'excluded_stocks.csv',index=False)
    pd.concat(stock_daily).to_csv(dest/'stock_equity.csv.gz',index=False,compression='gzip')
    pd.DataFrame(stock_trades).to_csv(dest/'stock_trades.csv.gz',index=False,compression='gzip')
    pd.DataFrame(stock_orders).to_csv(dest/'stock_orders.csv.gz',index=False,compression='gzip')
    # Confirm no source bytes changed during this read-only run.
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v['sha256'] for p,v in manifest.items())
    print('DONE: input hashes unchanged',flush=True)


if __name__ == '__main__':
    main()
