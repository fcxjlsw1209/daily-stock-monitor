"""Annual five-year market-only Ridge adjustment for monthly reconstruction."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import itertools
import sys
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
O = P.parent
sys.path.insert(0, str(O / 'monthly_adaptive_core'))
import adaptive as a
e, old, policy, strict = a.e, a.old, a.policy, a.strict
from research import fit_weighted_ridge, recency_weights
read, sha = a.read, a.sha
CONFIGS = [dict(alpha=alpha, half_life=half) for alpha in [1., 10., 100.] for half in [0, 12]]


def dump(name, value):
    (P / name).write_text(old.json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda v: v.item() if isinstance(v, np.generic) else v) + '\n')


def initialize():
    assert not (P / 'PLAN.json').exists()
    dump('PLAN.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        task='Test annual ordinary and recent-weighted Ridge forecasts as <=10% relative-ranking corrections to long-momentum monthly models.',
        data='Weekly samples in prior5calendar years; target next21session excess return vsSPY clipped[-1,1]. A sample enters training only when label end<=annual cutoff. No author holdings or trade outcomes as Ridge targets.',
        universe='Before2021 inherited historical proxy; thereafter most recent already-known reconstructed Top35 plus theme17. Label-end missing prices exclude only training examples, never select future tradable stocks.',
        features='Same32 strict-long-core/risk/shape/own-sector features as adaptive-core study. No recent stock-return factor or pullback gate. Include six observed market-style scalars and highvol/growth interactions.',
        configs=CONFIGS, cutoff='For each holding year2021..2026, cutoff is the preceding December monthend signal. All ordinary/recent Ridge training labels must have matured at cutoff.',
        seeds='Six distinct (pool,norm,core-menu,path-mode,context) specifications ranked only by adaptive study inner18-month matching; parameters re-fit in each fold. No prior outer result used for seed selection.',
        inverse_fit='6seeds x7forecast choices(including none) x4retention=168 policies, inner fit17/eval6 then fit23/eval12. ML coefficient constrained0..10%, core>=50%, other budgets unchanged. Fit only first35 author choices.',
        freeze='Choose overall and ML-category leader from inner18, re-fit author-ranking coefficients using35 months, freeze then validate34. Annual market-return forecast retraining in validation uses only then-past market data, not author labels.',
        execution='Zero cost, monthend final-close/same-close idealization, no intramonth trade; next-open control separate. Reused historical validation is not pristine OOS.',
        protections='Do not change original paper accounts, source data, past frozen candidates, or automate promotion.'))
    prev = read(a.P / 'PROTECTED_BEFORE.json')
    dump('PROTECTED_BEFORE.json', {f: sha(f) for f in prev})


def matured_rows(dates, ends, cutoff):
    start = str((pd.Timestamp(cutoff) - pd.DateOffset(years=5)).date())
    return [i for i, (d, end) in enumerate(zip(dates, ends)) if start <= d <= cutoff and end <= cutoff]


def prepare():
    px, raw, members, periods, sources, pools = e.load_prices()
    meta = read(a.P / 'INPUTS.json')
    names = meta['packs']['top35_percentile']['names']
    f, _, states = e.previous.panels(px, raw)
    core, side = e.features(px, raw)
    clean = {k: f[k] if k in f else core[k] if k in core else side[k]
             for k in names if k not in ['stock_sector_skip42_126', 'stock_sector_skip42_252', 'highvol_cmf63', 'trend_cmf63']}
    relatives = {f'stock_sector_skip42_{n}': px.shift(42) / px.shift(n) - 1 for n in [126, 252]}
    sf = O / 'index_union/alias_resolved_inputs/sectors.csv.gz'
    sectors = pd.read_csv(sf, index_col=0, parse_dates=True, low_memory=False)
    sources[str(sf)] = sha(sf)
    weekly = px.loc['2015-01-01':'2025-12-31'].index.to_series().groupby(px.loc['2015-01-01':'2025-12-31'].index.to_period('W-FRI')).last().tolist()
    monthly = pd.to_datetime([p['start'] for p in periods]).tolist()
    all_dates = sorted(set(weekly + monthly))
    style = old.market_styles(px, all_dates)
    warm = px.notna().rolling(295).sum().eq(295) & px.gt(0)
    mom = px.shift(42) / px.shift(252) - 1
    pooldates = pd.to_datetime(sorted(pools['35']))
    original_member = members['theme17'][0]
    F = len(names) * 3 + 6

    def row(date, ss):
        bm = strict.benchmark_map(date, ss, sectors, members['theme17'][1])
        tab = pd.DataFrame({k: v.loc[date, ss] for k, v in clean.items()})
        for k, v in relatives.items():
            tab[k] = strict.relative_row(v, date, ss, bm)
        q = e.engine.normalize(tab, 'percentile')
        q['highvol_cmf63'] = q.cmf63 * states.at[date, 'high_market_vol']
        q['trend_cmf63'] = q.cmf63 * states.at[date, 'spy_trend']
        x = q[names].to_numpy()
        st = style.loc[date]
        scalars = np.array([float(st[k]) * 2 - 1 for k in ['growth', 'smallcap', 'defensive', 'bonds', 'spy_above200', 'highvol']])
        return np.column_stack([x, x * scalars[0], x * scalars[-1], np.tile(scalars, (len(ss), 1))])

    X = np.zeros((len(weekly), 52, F))
    Y = np.full((len(weekly), 52), np.nan)
    valid = np.zeros((len(weekly), 52), bool)
    origins, ends = [], []
    for i, date in enumerate(weekly):
        prior = pooldates[pooldates <= date]
        if len(prior):
            ss = sorted({e.actual(s) for s in pools['35'][str(prior[-1].date())]} | members['theme17'][1])
            origin = str(prior[-1].date())
        else:
            ss = sorted(original_member.columns[original_member.loc[date]])
            origin = 'pre2021_inherited_historical_proxy'
        end = px.index[px.index.get_loc(date) + 21]
        nn = len(ss)
        assert nn <= 52 and set(ss).issubset(px.columns)
        X[i, :nn] = row(date, ss)
        Y[i, :nn] = ((px.loc[end, ss] / px.loc[date, ss] - 1) - (px.at[end, 'SPY'] / px.at[date, 'SPY'] - 1)).clip(-1, 1)
        valid[i, :nn] = (warm.loc[date, ss] & mom.loc[date, ss].gt(0)).to_numpy() & np.isfinite(Y[i, :nn])
        origins.append(origin)
        ends.append(str(end.date()))
    event_arrays = {}
    for key in ['top32_percentile', 'top35_percentile']:
        d = meta['packs'][key]
        A = np.zeros((69, max(map(len, d['symbols'])), F))
        for i, ss in enumerate(d['symbols']):
            A[i, :len(ss)] = row(monthly[i], ss)
        event_arrays[key.split('_')[0]] = A
    np.savez_compressed(P / 'MARKET_TRAINING.npz', X=X, Y=Y, eligible=valid, **event_arrays)
    dump('MARKET_TRAINING.json', dict(dates=[str(d.date()) for d in weekly], ends=ends, pool_origins=origins,
                                    periods=periods, feature_count=F, raw_names=names, weekly_rows=len(weekly)))
    for fpath in [a.P / 'INPUTS.json', a.P / 'INNER_TRIALS.json', Path(__file__)]:
        sources[str(fpath)] = sha(fpath)
    dump('SOURCE_HASHES.json', sources)
    print('Prepared causal market-only training', X.shape, flush=True)


def forecast():
    m = read(P / 'MARKET_TRAINING.json')
    data = np.load(P / 'MARKET_TRAINING.npz')
    specs = read(a.P / 'INPUTS.json')
    predictions = {key: np.zeros((6, 69, data[key].shape[1])) for key in ['top32', 'top35']}
    fits = []
    for year in range(2021, 2027):
        indices = [i for i, p in enumerate(m['periods']) if pd.Timestamp(p['end']).year == year]
        cutoff = m['periods'][min(indices)]['start']
        ii = matured_rows(m['dates'], m['ends'], cutoff)
        XX, yy, ages = [], [], []
        for i in ii:
            good = data['eligible'][i]
            XX.append(data['X'][i, good])
            yy.append(data['Y'][i, good])
            ages += [(pd.Timestamp(cutoff) - pd.Timestamp(m['dates'][i])).days / 30.4375] * int(good.sum())
        X, y, age = np.vstack(XX), np.concatenate(yy), np.array(ages)
        latest = max(m['ends'][i] for i in ii)
        assert latest <= cutoff and len(X) > 1000
        for ci, cfg in enumerate(CONFIGS):
            fit = fit_weighted_ridge(X, y, recency_weights(age, cfg['half_life']), cfg['alpha'])
            for key in predictions:
                pred = np.einsum('ijk,k->ij', data[key][indices], fit['coef'], optimize=False) + fit['intercept']
                eligible = np.load(a.P / (key + '_percentile.npz'))['eligible']
                for j, i in enumerate(indices):
                    predictions[key][ci, i] = pd.Series(pred[j]).where(eligible[i]).rank(pct=True).fillna(.5).to_numpy() - .5
            fits.append(dict(year=year, configuration=ci, config=cfg, cutoff=cutoff, latest_label_end=latest,
                             training_start=m['dates'][ii[0]], rows=len(X), **{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fit.items()}))
        print('Annual forecasts', year, 'examples', len(X), 'labels through', latest, flush=True)
    np.savez_compressed(P / 'PREDICTIONS.npz', **predictions)
    dump('FORECAST_AUDIT.json', fits)


def append_forecast(X, spec, forecasts, index):
    if index < 0:
        return X, spec
    return np.concatenate([X, forecasts[index, :, :, None]], axis=-1), spec + [('annual_ridge', 1., 'ml')]


def train():
    m, packs, px, opening, basecontext, styles = a.load()
    forecasts = np.load(P / 'PREDICTIONS.npz')
    seeds, seen = [], set()
    for r in sorted(read(a.P / 'INNER_TRIALS.json'), key=a.choose, reverse=True):
        key = tuple(r[k] for k in ['key', 'menu', 'mode', 'partition'])
        if key not in seen:
            seeds.append(r)
            seen.add(key)
        if len(seeds) == 6:
            break
    dump('SEEDS.json', seeds)
    trials = []
    for si, seed in enumerate(seeds):
        d = packs[seed['key']]
        c = a.context_for(d, basecontext)
        baseX, basespec = a.design(d, a.CORE_MENUS[seed['menu']], seed['mode'])
        groups = styles[seed['partition']].to_numpy()
        for forecast_id in range(-1, 6):
            X, spec = append_forecast(baseX, basespec, forecasts['top' + str(seed['pool'])], forecast_id)
            ag = {(b, gap): [] for b, gap in old.RETENTION}
            for before, until in a.FOLDS:
                training = np.arange(69) < before
                val = (np.arange(69) >= before) & (np.arange(69) < until)
                rule = a.fit(X, spec, d, groups, training)
                scores, strength, local, W = a.scores_for(rule, X, spec, d, c, groups)
                for b, gap in old.RETENTION:
                    sim = policy.simulate(c, scores, strength, dict(buffer=b, gap=gap, exit=dict(name='none')), count=until)
                    ag[(b, gap)].append(policy.matching(sim['choices'], scores, c, val))
            for (b, gap), mm in ag.items():
                totals = {k: sum(z[k] for z in mm) for k in ['months', 'shared_slots', 'set_exact', 'order_exact']}
                trials.append(dict(id=len(trials), seed=si, forecast_id=forecast_id,
                    config=dict(buffer=b, gap=gap, exit=dict(name='none')), inner_validation=totals,
                    **{k: seed[k] for k in ['key', 'pool', 'menu', 'mode', 'partition']}))
        print('Ridge policy comparisons', len(trials), flush=True)
    key = lambda r: (a.choose(r)[:-1], -(r['forecast_id'] >= 0), -r['id'])
    selected = {n: max(rows, key=key) for n, rows in [('overall', trials), ('ml', [r for r in trials if r['forecast_id'] >= 0])]}
    frozen = {}
    for name, r in selected.items():
        d = packs[r['key']]
        X, spec = a.design(d, a.CORE_MENUS[r['menu']], r['mode'])
        X, spec = append_forecast(X, spec, forecasts['top' + str(r['pool'])], r['forecast_id'])
        rule = a.fit(X, spec, d, styles[r['partition']].to_numpy(), np.arange(69) < 35)
        frozen[name] = dict(**r, rule=rule)
    dump('INNER_TRIALS.json', trials)
    dump('FROZEN.json', dict(created_at=datetime.now(timezone.utc).isoformat(), leaders=frozen, train35_validate34=True))
    print('FROZEN', {n: {k: r[k] for k in ['key', 'menu', 'partition', 'forecast_id', 'inner_validation']} for n, r in frozen.items()}, flush=True)


def evaluate():
    h = sha(P / 'FROZEN.json')
    assert all(sha(f) == v for f, v in read(P / 'SOURCE_HASHES.json').items())
    m, packs, px, opening, basecontext, styles = a.load()
    forecasts = np.load(P / 'PREDICTIONS.npz')
    results = {}
    for name, r in read(P / 'FROZEN.json')['leaders'].items():
        d = packs[r['key']]
        c = a.context_for(d, basecontext)
        X, spec = a.design(d, a.CORE_MENUS[r['menu']], r['mode'])
        X, spec = append_forecast(X, spec, forecasts['top' + str(r['pool'])], r['forecast_id'])
        scores, strength, local, W = a.scores_for(r['rule'], X, spec, d, c, styles[r['partition']].to_numpy())
        sim = policy.simulate(c, scores, strength, r['config'], detailed=True)
        met, slices, eq, trades = old.performance(px, opening, c, sim, scores, 'close')
        os = policy.simulate(c, scores, strength, r['config'], execution='open')
        om, oss, _, _ = old.performance(px, opening, c, os, scores, 'open')
        matching = {k: policy.matching(sim['choices'], scores, c, mask) for k, mask in old.masks(69).items() if k in ['training', 'validation', 'all']}
        results[name] = dict(metrics=met, slices=slices, next_open=om, next_open_slices=oss, matching=matching, rule=r)
        dump(name + '_TRADES.json', trades)
        dump(name + '_DECISIONS.json', sim['events'])
        dump(name + '_DAILY.json', [dict(date=str(t.date()), equity=float(v)) for t, v in eq.items()])
        print(name, dict(full=met, validation=slices['validation34'], matching=matching), flush=True)
    dump('RESULTS.json', results)
    assert sha(P / 'FROZEN.json') == h
    assert all(z['latest_label_end'] <= z['cutoff'] for z in read(P / 'FORECAST_AUDIT.json'))
    changes = [f for f, v in read(P / 'PROTECTED_BEFORE.json').items() if sha(f) != v]
    assert not changes, changes
    dump('VALIDATION.json', dict(protected_changes=changes, frozen_hash=h, source_hashes_verified=True,
        annual_labels_matured=True, no_author_labels_in_market_forecasts=True, no_outer_labels_in_inverse_fit=True,
        zero_cost=True, fixed_monthly_units=True, independent_replay=True, not_pristine_oos=True))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['initialize', 'prepare', 'forecast', 'train', 'evaluate'])
    globals()[ap.parse_args().phase]()
