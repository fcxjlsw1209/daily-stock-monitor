"""Daily pullback episode memory as a zero-default monthly boundary correction.

Research only. Never writes an observer, trade plan, or existing source file.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import itertools
import json
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parent
O = P.parent
sys.path.insert(0, str(O / 'monthly_company_context'))
import context_research as previous

old, prior, rp, ex, read, sha = previous.old, previous.prior, previous.rp, previous.ex, previous.read, previous.sha


def dump(name, value):
    (P / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else x) + '\n')


def models():
    return [dict(id=i, horizon=h, match=match, target=target, mode=mode)
            for i, (h, match, target, mode) in enumerate(itertools.product(
                [5, 10, 21], ['signal', 'context'], ['excess', 'utility'], ['level', 'increment']))]


def grid():
    rows = [dict(id=0, family='original65'), dict(id=1, family='previous66')]
    for model, budget in itertools.product(models(), [.025, .05]):
        rows.append(dict(id=len(rows), family=model['mode'], model_id=model['id'],
                         budget=budget, ceiling=5, gap=.02))
    return rows


def initialize():
    assert not (P / 'PLAN.json').exists()
    dump('PLAN.json', dict(
        purpose='Test daily non-overlapping same-security rebound episodes, with exactly zero ranking change when evidence is absent.',
        fixed='Same122 archived Top32-estimate+theme17 monthly pools, original eligibility/skip42 momentum, long only threeequal/twohalf/<2cash, fixed monthly units. 65.34 control and frozen66.10 baseline unchanged.',
        episode='Use cached historical prices for each security appearing in the monthly research pools. Security may contribute its OWN earlier history outside the trading Top32; this does not expand tradable pools. Positive252skip42 momentum,295 complete prices, causal finite features. A=five-day return below minus0.5ATR21/close; B=close-location<=.25. Aonly/Bonly/both as prior study proxies, not claimed author formulas. Greedy first event then21-session lockout independent of outcomes; future missing labels never alter anchor selection.',
        labels='Entry at next adjusted open, exits at5/10/21-session adjusted close. Excess versus SPY over same entry/exit; utility=excess-0.5*own maximum adverse CLOSE excursion. Require entire path and benchmark finite. Outcomes mature by signal close; no query event predicts itself. Same-security21-session label windows do not overlap; events across securities are not statistically independent.',
        features='Eight bounded own-state features: skipped long momentum, risk-adjusted momentum,63-session volatility,5-day ATR-normalized decline,close-location,CMF21,trend efficiency and signed volume63. Three causal market states:SPY200trend,SPY63annualvol,andQQQ-relativeSPY126return. No earnings/news dates or author labels.',
        estimation='At each query month-end use only prior5years with label_end<=query. Exact same security and signal type.24month decay; context match adds Gaussian own-shape bandwidth0.5 and market-context bandwidth1. Prior mass6 shrinks to ZERO increment. Require effective_samples>=3. Subtract one estimated standard error from absolute predicted effect, floored0. No evidence/no current signal means exact zero, not fallback globalRidge.',
        effect='Level estimates conditional excess/utility. Increment estimates conditional outcome minus unconditional own-event outcome; conservative SE sqrt(var_cond/neff_cond+var_all/neff_all). Map shrunk effect to0.5*tanh(effect/max(vol63*sqrt(horizon),.01)). No new ML is fitted; prior candidate retains its frozen annual Ridge component.',
        score='Keep first2 names. Only original ranks3..5 within.02 of third compete. Set new support target=clip(originalsupport+effect,-.5,.5), then add budget*(target-originalsupport). Budget2.5/5%, total old+new support corrections<=10%; long core untouched. Zero effect is algebraically zero change. No intramonth stops or turnover buffer.',
        grid='24 models=3 horizons x2 matching modes x2 labels x2 effects; each2budgets=48new plus2controls. Fixed before any performance comparisons.',
        selection='First88 through2023-11-30 only; same guards relative66.10: trainSharpe>=base-.05,DD>=base-.03,early53CAGR>=base,recent35CAGR>=base-.02. Freeze best admitted primary and worst-segment robust plus best admitted effect-family; otherwise retain base. No rejected candidate later34 replay. Sequential36/60/76 diagnostic within training.',
        reporting='Same-final-close zero-cost idealization, next-openzero and5bps/side sensitivity; year/subperiod returns, changed-month attribution, sample counts and no-history rates. Repeatedly researched history is not pristine OOS. No guarantee of improvement or original-author reproduction.',
        protection='No old code/config, rules, pool, plans, ledgers, automation or candidate registration changes. New files only in this folder.',
        created_utc=datetime.now(timezone.utc).isoformat()))
    dump('MODEL_CONFIGS.json', models())
    dump('GRID.json', grid())
    protected = read(previous.P / 'PROTECTED.json')
    protected.update({str(f): sha(f) for f in previous.P.rglob('*') if f.is_file() and f.suffix != '.pyc'})
    dump('PROTECTED.json', protected)
    files = [Path(previous.__file__), Path(old.__file__), Path(rp.__file__), Path(ex.e.engine.__file__),
             old.P / 'FROZEN.json', old.P / 'RESULTS.json']
    dump('INPUT_HASHES.json', {str(f): sha(f) for f in files})


def load():
    for path, h in read(P / 'INPUT_HASHES.json').items():
        assert sha(path) == h, path
    m, d, _, _, px, opening, _, a = old.load()
    return m, d, px, opening, a


def anchors(signal, eligible, lockout=21):
    out = []
    next_allowed = 0
    for i in range(len(signal)):
        if i >= next_allowed and signal[i] > 0 and eligible[i]:
            out.append(i)
            next_allowed = i + lockout
    return np.array(out, dtype=int)


def label_path(px, opening, market, market_open, index, horizon):
    end = index + horizon
    if end >= len(px):
        return np.nan, np.nan
    path = np.asarray(px[index + 1:end + 1], float)
    mp = np.asarray(market[index + 1:end + 1], float)
    entry, me = opening[index + 1], market_open[index + 1]
    if not (np.isfinite(path).all() and np.isfinite(mp).all() and (path > 0).all()
            and (mp > 0).all() and np.isfinite(entry) and entry > 0 and np.isfinite(me) and me > 0):
        return np.nan, np.nan
    returns = path / entry - 1
    excess = returns[-1] - (mp[-1] / me - 1)
    adverse = max(0., -float(returns.min()))
    return float(np.clip(excess, -1, 1)), float(np.clip(excess - .5 * adverse, -1, 1))


def features(panel):
    px, close, high, low, volume = [panel[k] for k in ['adj', 'close', 'high', 'low', 'volume']]
    r = px.pct_change(fill_method=None)
    vol = r.rolling(63).std()
    span = (high - low).replace(0, np.nan)
    tr = pd.DataFrame(np.maximum.reduce([span.to_numpy(), (high - close.shift()).abs().to_numpy(),
                                         (low - close.shift()).abs().to_numpy()]), index=px.index, columns=px.columns)
    atr = tr.rolling(21).mean() / close
    mom = px.shift(42) / px.shift(252) - 1
    loc = (2 * close - high - low) / span
    cmf = (loc * volume).rolling(21).sum() / volume.rolling(21).sum().replace(0, np.nan)
    lr = np.log(px / px.shift())
    efficiency = np.log(px.shift(42) / px.shift(294)) / lr.abs().rolling(252).sum().shift(42).replace(0, np.nan)
    signed = (np.sign(r) * volume).rolling(63).sum() / volume.rolling(63).sum().replace(0, np.nan)
    shape = [np.tanh(mom), np.tanh(mom / (vol * np.sqrt(252)).replace(0, np.nan)),
             (vol * np.sqrt(252)).clip(0, 1), np.tanh(px.pct_change(5, fill_method=None) / atr / 3),
             loc.clip(-1, 1), cmf.clip(-1, 1), efficiency.clip(-1, 1), signed.clip(-1, 1)]
    market = pd.DataFrame(dict(
        trend=(2 * (px.SPY > px.SPY.rolling(200).mean()).astype(float) - 1).where(px.SPY.rolling(200).mean().notna()),
        volatility=(vol.SPY * np.sqrt(252) / .30).clip(0, 2) - 1,
        growth=np.tanh((px.QQQ.pct_change(126, fill_method=None) - px.SPY.pct_change(126, fill_method=None)) / .20)))
    signal = previous.signal_frames(panel)
    good = px.notna().rolling(295).sum().eq(295) & mom.gt(0)
    for f in shape:
        good &= np.isfinite(f)
    good &= market.notna().all(axis=1).to_numpy()[:, None]
    return shape, market, signal, good, vol


def weighted(y, w):
    total = float(w.sum())
    if total <= 0:
        return 0., 0., 0., 0.
    mean = float(w @ y / total)
    variance = float(w @ (y - mean) ** 2 / total)
    effective = total ** 2 / float(w @ w)
    return mean, variance, effective, total


def estimate(events, query, cfg):
    """All filters depend on past dates and known query features, never future Y."""
    none = dict(observations=0, effective_samples=0., own_weight=0., latest_label_end=None,
                raw_effect=0., standard_error=0., shrunk_effect=0.)
    if query['signal'] <= 0 or not query['eligible']:
        return 0., none
    h = cfg['horizon']
    cutoff = query['date']
    earliest = cutoff - np.timedelta64(1827, 'D')
    ends = events['ends_' + str(h)]
    labels = events[cfg['target'] + '_' + str(h)]
    mask = ((events['symbol'] == query['symbol']) & (events['dates'] >= earliest)
            & (events['dates'] < cutoff) & (ends <= cutoff) & np.isfinite(labels))
    ids = np.flatnonzero(mask)
    if not len(ids):
        return 0., none
    dates = events['dates'][ids]
    age_months = (cutoff - dates).astype('timedelta64[D]').astype(float) / 30.4375
    all_weights = 2 ** (-age_months / 24.)
    weights = all_weights * (events['signal'][ids] == query['signal'])
    if cfg['match'] == 'context':
        ds = np.mean((events['shape'][ids] - query['shape']) ** 2, axis=1)
        dm = np.mean((events['market'][ids] - query['market']) ** 2, axis=1)
        weights *= np.exp(-.5 * ds / .5 ** 2 - .5 * dm)
    used = weights > 0
    n = int(used.sum())
    mean, variance, effective, mass = weighted(labels[ids], weights)
    info = dict(none, observations=n, effective_samples=effective,
                latest_label_end=str(ends[ids[used]].max()) if n else None)
    if effective < 3:
        return 0., info
    se = np.sqrt(variance / effective)
    effect = mean
    if cfg['mode'] == 'increment':
        unconditional, uv, un, _ = weighted(labels[ids], all_weights)
        effect -= unconditional
        se = np.sqrt(variance / effective + uv / un)
    confidence = mass / (mass + 6.)
    shrunk = np.sign(effect) * max(abs(effect) - se, 0.) * confidence
    delta = .5 * np.tanh(shrunk / max(query['vol'] * np.sqrt(h), .01))
    info.update(own_weight=confidence, raw_effect=float(effect), standard_error=float(se), shrunk_effect=float(shrunk))
    return float(delta), info


def prepare():
    m, d, px, op, a = load()
    panel, info = old.panel_load(m['etfs'])
    symbols = sorted(set(sum(m['symbols'], [])))
    needed = sorted(set(symbols) | {'SPY', 'QQQ'})
    panel = {k: v.reindex(columns=needed).loc[:m['ends'][-1]].copy()
             for k, v in panel.items() if k in ['adj', 'close', 'high', 'low', 'volume', 'open']}
    shape, market, signal, good, vol = features(panel)
    dates = panel['adj'].index.to_numpy(dtype='datetime64[D]')
    normalized_open = panel['adj'] * panel['open'] / panel['close']
    ev = {k: [] for k in ['symbol', 'dates', 'anchor_index', 'signal', 'shape', 'market']}
    for h in [5, 10, 21]:
        for kind in ['ends', 'excess', 'utility']:
            ev[kind + '_' + str(h)] = []
    spy = panel['adj'].SPY.to_numpy()
    spy_open = normalized_open.SPY.to_numpy()
    for s in symbols:
        starts = anchors(signal[s].to_numpy(), good[s].to_numpy())
        for idx in starts:
            ev['symbol'].append(s)
            ev['dates'].append(dates[idx])
            ev['anchor_index'].append(idx)
            ev['signal'].append(int(signal[s].iloc[idx]))
            ev['shape'].append([v[s].iloc[idx] for v in shape])
            ev['market'].append(market.iloc[idx].to_numpy())
            for h in [5, 10, 21]:
                y, u = label_path(panel['adj'][s].to_numpy(), normalized_open[s].to_numpy(), spy, spy_open, idx, h)
                ev['ends_' + str(h)].append(dates[idx + h] if idx + h < len(dates) else np.datetime64('2262-01-01'))
                ev['excess_' + str(h)].append(y)
                ev['utility_' + str(h)].append(u)
    events = {k: np.array(v) for k, v in ev.items()}
    np.savez_compressed(P / 'EVENTS.npz', **events)
    n, width = d['eligible'].shape
    query = dict(shape=np.zeros((n, width, 8)), market=np.zeros((n, 3)),
                 signal=np.zeros((n, width), int), vol=np.zeros((n, width)))
    for i, (date, ss) in enumerate(zip(m['dates'], m['symbols'])):
        query['shape'][i] = np.column_stack([f.loc[date, ss] for f in shape])
        query['market'][i] = market.loc[date]
        query['signal'][i] = signal.loc[date, ss].to_numpy(int)
        query['vol'][i] = vol.loc[date, ss]
    np.savez_compressed(P / 'QUERIES.npz', **query)
    forecasts = {str(c['id']): np.zeros((n, width)) for c in models()}
    details = []
    base = old.score(read(old.P / 'FROZEN.json')['selected']['primary']['spec'], a, n)
    for i, (date, ss) in enumerate(zip(m['dates'], m['symbols'])):
        ranks = sorted(np.flatnonzero(d['eligible'][i]), key=lambda j: (-base[i, j], ss[j]))
        boundary = [j for j in ranks[2:5] if base[i, ranks[2]] - base[i, j] <= .02] if len(ranks) >= 3 else []
        for j, symbol in enumerate(ss):
            q = dict(symbol=symbol, date=np.datetime64(date, 'D'), eligible=bool(d['eligible'][i, j]),
                     shape=query['shape'][i, j], market=query['market'][i],
                     signal=query['signal'][i, j], vol=query['vol'][i, j])
            if not (np.isfinite(q['shape']).all() and np.isfinite(q['market']).all() and np.isfinite(q['vol'])):
                q['eligible'] = False
            for cfg in models():
                value, detail = estimate(events, q, cfg)
                forecasts[str(cfg['id'])][i, j] = value
                if j in boundary:
                    details.append(dict(date=date, symbol=symbol, model_id=cfg['id'], signal_type=int(q['signal']),
                                        split='training' if i < 88 else 'later34', correction=value, **detail))
        if i % 12 == 0:
            print('Forecasted', date, flush=True)
    # Causal feature construction must be identical when the future is absent.
    cut = pd.Timestamp('2020-12-31')
    short = features({k: v.loc[:cut].copy() for k, v in panel.items()})
    for full, trimmed in zip(shape, short[0]):
        pd.testing.assert_frame_equal(full.loc[:cut], trimmed)
    for full, trimmed in [(market, short[1]), (signal, short[2]), (good, short[3]), (vol, short[4])]:
        pd.testing.assert_frame_equal(full.loc[:cut], trimmed)
    np.savez_compressed(P / 'FORECASTS.npz', **forecasts)
    dump('BOUNDARY_DETAILS.json', details)
    dump('DATA_AUDIT.json', dict(episodes=len(events['dates']), securities=len(symbols),
        earliest=str(events['dates'].min()), latest=str(events['dates'].max()),
        valid_labels={str(h): int(np.isfinite(events['excess_' + str(h)]).sum()) for h in [5, 10, 21]},
        source_hashes=info['source_hashes'], prefix_features_equal=True))
    print('Prepared', len(events['dates']), 'episodes across', len(symbols), 'securities', flush=True)


def select_boundary(base, eligible, symbols, effects, a, row):
    n = len(base)
    score = base.copy()
    support = (a['baseline'][:n] - a['core'][:n]) / (1 - a['core_weight'][:n])
    oldbudget = .025 * a['state_calm10'][:n] + .05 * a['state_weak_narrow'][:n]
    budget = np.minimum(row['budget'], np.maximum(0., .10 - oldbudget))
    choices = []
    for i, ss in enumerate(symbols):
        rank = sorted(np.flatnonzero(eligible[i]), key=lambda j: (-base[i, j], ss[j]))
        if len(rank) < 3:
            choices.append(rank if len(rank) == 2 else [])
            continue
        candidates = [j for j in rank[2:row['ceiling']] if base[i, rank[2]] - base[i, j] <= row['gap']]
        for j in candidates:
            target = np.clip(support[i, j] + effects[i, j], -.5, .5)
            score[i, j] += budget[i] * (target - support[i, j])
        winner = min(candidates, key=lambda j: (-score[i, j], ss[j]))
        choices.append(sorted(rank[:2] + [winner], key=lambda j: (-score[i, j], ss[j])))
    return choices, score


def choices(row, m, d, a, pred, n):
    if row['family'] == 'original65':
        score = a['baseline'][:n].copy()
    else:
        score = old.score(read(old.P / 'FROZEN.json')['selected']['primary']['spec'], a, n)
    if row['id'] < 2:
        return rp.choose(score, d['eligible'][:n], m['symbols'][:n], 3), score
    return select_boundary(score, d['eligible'][:n], m['symbols'][:n], pred[str(row['model_id'])][:n], a, row)


def measure(row, m, d, a, pred, blocks, bounds, valid):
    ch, _ = choices(row, m, d, a, pred, 88)
    rr = rp.fast_returns(ch, blocks, valid)
    train = rp.stats_rr(rr, m['dates'][0], m['ends'][87])
    early = rp.stats_rr(rr[:bounds[53]], m['dates'][0], m['ends'][52])
    recent = rp.stats_rr(rr[bounds[53]:], m['dates'][53], m['ends'][87])
    robust = min(early['sharpe'], recent['sharpe']) + .3 * train['sharpe'] + .4 * np.log1p(train['cagr']) - .3 * abs(train['max_drawdown'])
    return dict(row, training=train, early=early, recent=recent, robust=float(robust)), rr


def train():
    assert not (P / 'FROZEN.json').exists()
    m, d, px, op, a = load()
    pred = dict(np.load(P / 'FORECASTS.npz'))
    blocks, _, bounds, valid = rp.paths(px, m, 88)
    rows, returns = [], []
    for row in read(P / 'GRID.json'):
        z, rr = measure(row, m, d, a, pred, blocks, bounds, valid)
        rows.append(z)
        returns.append(rr)
    admitted = [r for r in rows if prior.allowed(r, rows[1])]
    pick = lambda group: max(group, key=lambda r: (r['training']['cagr'], -r['id']))
    family = {k: [r for r in admitted if r['family'] == k] for k in ['level', 'increment']}
    selected = dict(primary=pick(admitted), robust=max(admitted, key=lambda r: (r['robust'], -r['id'])))
    controls = {k: pick(v) if v else rows[1] for k, v in family.items()}
    wf, stitched, baseparts = [], [], []
    for n, end in [(36, 60), (60, 76), (76, 88)]:
        metrics = [rp.stats_rr(rr[:bounds[n]], m['dates'][0], m['ends'][n - 1]) for rr in returns]
        b = metrics[1]
        ids = [i for i, v in enumerate(metrics) if v['sharpe'] >= b['sharpe'] - .05 and v['max_drawdown'] >= b['max_drawdown'] - .03]
        chosen = max(ids, key=lambda i: (metrics[i]['cagr'], -i))
        rr, br = returns[chosen][bounds[n]:bounds[end]], returns[1][bounds[n]:bounds[end]]
        stitched.extend(rr)
        baseparts.extend(br)
        wf.append(dict(train_end=m['ends'][n - 1], test_start=m['dates'][n], test_end=m['ends'][end - 1],
                       chosen_id=chosen, metrics=rp.stats_rr(rr, m['dates'][n], m['ends'][end - 1]),
                       baseline=rp.stats_rr(br, m['dates'][n], m['ends'][end - 1])))
    dump('WALKFORWARD.json', dict(folds=wf, metrics=rp.stats_rr(np.array(stitched), m['dates'][36], m['ends'][87]),
        baseline=rp.stats_rr(np.array(baseparts), m['dates'][36], m['ends'][87]), pristine_oos=False))
    dump('TRAIN_TRIALS.json', rows)
    np.savez_compressed(P / 'TRAIN_RETURNS.npz', returns=np.stack(returns), bounds=np.array(bounds))
    frozen = dict(created_utc=datetime.now(timezone.utc).isoformat(), selected=selected,
        family_controls=controls, family_admissible={k: len(v) for k, v in family.items()},
        original=rows[0], baseline=rows[1], trials=len(rows), code_sha256=sha(__file__),
        input_hashes={f: sha(P / f) for f in ['PLAN.json', 'GRID.json', 'MODEL_CONFIGS.json', 'INPUT_HASHES.json',
                                            'EVENTS.npz', 'QUERIES.npz', 'FORECASTS.npz', 'DATA_AUDIT.json']})
    dump('FROZEN.json', frozen)
    (P / 'FROZEN.sha256').write_text(sha(P / 'FROZEN.json') + '\n')
    print('Frozen', [(k, v['id'], v['training']['cagr']) for k, v in selected.items()], frozen['family_admissible'], flush=True)


def evaluate():
    frozen = read(P / 'FROZEN.json')
    assert sha(P / 'FROZEN.json') == (P / 'FROZEN.sha256').read_text().strip()
    assert sha(__file__) == frozen['code_sha256']
    for name, h in frozen['input_hashes'].items():
        assert sha(P / name) == h, name
    m, d, px, op, a = load()
    pred = dict(np.load(P / 'FORECASTS.npz'))
    out, allorders = {}, {}
    blocks, _, bounds, valid = rp.paths(px, m, len(m['dates']))
    configs = dict(original65=frozen['original'], previous66=frozen['baseline'],
                   **frozen['family_controls'], **frozen['selected'])
    for name, row in configs.items():
        ch, sc = choices(row, m, d, a, pred, len(m['dates']))
        orders = rp.make_orders(ch, sc, m)
        met, eq, trades, holdings = ex.e.engine.replay(px, op, orders, cost=0., execution='close',
            start=pd.Timestamp(m['dates'][0]), end=pd.Timestamp(m['ends'][-1]))
        rr = rp.fast_returns(ch, blocks, valid)
        np.testing.assert_allclose(eq, np.r_[1., np.cumprod(1 + rr)] * 100000, rtol=1e-11, atol=1e-5)
        cuts = prior.old.previous.cuts(eq, m)
        assert abs(cuts['training2016_2023']['cagr'] - row['training']['cagr']) < 1e-10
        out[name] = dict(metrics=met, slices=cuts, configuration=row)
        if name in ['original65', 'previous66']:
            ref = read(old.P / 'RESULTS.json')['baseline' if name == 'original65' else 'primary']['metrics']
            for key in ['cagr', 'sharpe', 'max_drawdown']:
                assert abs(met[key] - ref[key]) < 1e-10
        for label, cost in [('next_open_zero', 0.), ('next_open_cost', .0005)]:
            mm, ee, _, _ = ex.e.engine.replay(px, op, orders, cost=cost, execution='open',
                start=pd.Timestamp(m['dates'][0]), end=pd.Timestamp(m['ends'][-1]))
            out[name][label] = dict(metrics=mm, slices=prior.old.previous.cuts(ee, m))
        for suffix, value in [('ORDERS', orders), ('TRADES', trades), ('HOLDINGS', holdings),
                              ('DAILY', [dict(date=str(t.date()), equity=float(v)) for t, v in eq.items()])]:
            dump(name + '_' + suffix + '.json', value)
        allorders[name] = orders
        print(name, 'configuration', row['id'], 'CAGR', met['cagr'], 'Sharpe', met['sharpe'], flush=True)
    q = px.loc[m['dates'][0]:m['ends'][-1], 'QQQ']
    eq = q / q.iloc[0] * 100000
    out['qqq'] = dict(metrics=ex.e.engine.stats(eq), slices=prior.old.previous.cuts(eq, m))
    dump('RESULTS.json', out)
    comparisons = []
    for i, day in enumerate(m['dates']):
        line = dict(date=day, end=m['ends'][i], split='early53' if i < 53 else 'recent35' if i < 88 else 'later34')
        for name, orders in allorders.items():
            ss = orders[i]['symbols']
            line[name] = dict(symbols=ss, monthly_return=float((px.loc[m['ends'][i], ss] / px.loc[day, ss] - 1).mean()) if ss else 0.)
        comparisons.append(line)
    dump('MONTH_COMPARISON.json', comparisons)
    bad = [p for p, h in read(P / 'PROTECTED.json').items() if sha(p) != h]
    assert not bad, bad
    dump('VALIDATION.json', dict(independent_daily_replay_parity=True, controls122_parity=True,
        frozen_training_parity=True, protected_files=len(read(P / 'PROTECTED.json')), protected_changes=bad,
        no_observer_promoted=True, pristine_oos=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['initialize', 'prepare', 'train', 'evaluate'])
    globals()[parser.parse_args().phase]()
