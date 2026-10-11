"""Constrained inverse attribution, NOT an executable trading strategy.

Use the author's published path, Sharpe, drawdown and QQQ correlation together.
No ledger or old research writes. Intermediate selections are latent search
states, not admitted potential author transactions.
"""
from pathlib import Path
from itertools import combinations
import sys, json, hashlib, argparse
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
O = P.parent
I = O / 'monthly_curve_reverse'
R = O / 'monthly_reconstruction_v2'
sys.path.insert(0, str(I))
from inverse import interpolation_matrix, score_paths, load_observations, mul

START = pd.Timestamp('2016-08-17')
END = pd.Timestamp('2026-09-24')
TARGET = dict(wealth=387.72, cagr=.8064, sharpe=1.867, max_dd=-.3446, correlation=.712)


def sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def read(f):
    return json.loads(Path(f).read_text())


def dump(name, obj):
    (P / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def load():
    source = [O / 'index_union/alias_resolved_inputs/adj.csv.gz',
              O / 'daily_rebuild_research/price_scaled_membership.csv.gz',
              O / 'four_factor_rebound/ETF_SOURCES.json', R / 'ETF_MANIFEST.json']
    px = pd.read_csv(source[0], index_col=0, parse_dates=True)
    old = read(source[2]); extra = read(source[3])
    for row in old + extra:
        assert sha(row['file']) == row['sha256']
        px[row['symbol']] = pd.read_csv(row['file'], index_col=0, parse_dates=True)['Adj Close'].reindex(px.index)
        source.append(Path(row['file']))
    member = pd.read_csv(source[1], index_col=0, parse_dates=True).reindex(index=px.index, columns=px.columns, fill_value=False).fillna(False).astype(bool)
    etfs = {v['symbol'] for v in old + extra} | {'TQQQ', 'QLD'}
    member.loc[:, sorted(etfs & set(member.columns))] = False
    core = [r['symbol'] for r in old]
    theme = [s for s in core if s not in {'DIA', 'XLB', 'XLC', 'XLRE', 'XLU', 'XLP'}] + ['SMH', 'SOXX', 'IGV', 'VGT', 'MTUM', 'ARKK']
    pools = {}
    for label, etf in [('core17', core), ('theme17', theme)]:
        pools[label] = member.copy()
        pools[label][etf] = px[etf].notna()
    sources = {str(f): sha(f) for f in source}
    sources.update({str(I / f): sha(I / f) for f in ['CALIBRATION.json', 'DIGITIZED_WEALTH.csv', 'purple_dd_pixels.csv', 'SOURCE.jpg']})
    return px, pools, sources


def xcoord(index, cal):
    a = cal['wealth']
    return a['x0'] + a['width'] * (index - START).days.to_numpy() / (END - START).days


def dates_and_events(px):
    dates = px.loc[START:END].index
    events = [dates[0]] + [d for d in dates[1:] if d != END and px.index[px.index.get_loc(d)+1].to_period('M') != d.to_period('M')]
    return dates, events


def prepare(pool):
    px, pools, sources = load()
    cal = read(I / 'CALIBRATION.json'); obs = load_observations()
    member = pools[pool]
    mom = px.shift(42) / px.shift(252) - 1
    good = member & mom.gt(0) & px.notna().rolling(253, min_periods=253).sum().eq(253) & px.gt(0)
    dates, events = dates_and_events(px)
    warm = read(R / f'INVERSE_{pool}_q0.json')['orders']
    warm = {row['date']: set(row['holdings']) for row in warm}
    folder = P / 'cache'; folder.mkdir(exist_ok=True)
    metadata = []
    for k, start in enumerate(events):
        end = events[k+1] if k+1 < len(events) else END
        f = px.loc[start:end]
        cols = sorted(t for t in px if bool(good.at[start, t]) and f[t].notna().all() and f[t].gt(0).all())
        # Future completeness is inverse-attribution feasibility only, not an entry filter.
        n = min(3, len(cols))
        combos = np.asarray(list(combinations(range(len(cols)), n)), dtype=np.int16) if n >= 2 else np.zeros((1, 0), dtype=np.int16)
        ratios = (f[cols] / f[cols].iloc[0]).to_numpy()
        paths = ratios[:, combos].mean(axis=2).T if n >= 2 else np.ones((1, len(f)))
        logs = np.log(paths); r = paths[:, 1:] / paths[:, :-1] - 1
        q = px.loc[start:end, 'QQQ'].pct_change().dropna().to_numpy()
        assert len(q) == r.shape[1]
        x = xcoord(f.index, cal)
        ob = {p: z[(z.x > x[0] + .35) & (z.x < x[-1] - .35)] for p, z in obs.items()}
        # Initial boundary uses the visible relative panel before wealth pixels become available.
        reference = obs['relative'] if x[0] < obs['wealth'].x.min() else obs['wealth']
        anchor = 0. if k == 0 else float(np.interp(x[0], reference.x, reference.log_wealth))
        previous = pd.concat([z[z.x <= x[0]] for p, z in obs.items() if p != 'dd'])
        peak = max(0., anchor, float(previous.log_wealth.max()) if len(previous) else 0.)
        constraint = dict(start_log_nav=anchor, prior_log_peak=peak,
                          log_tolerance=0. if k == 0 else 1.5 / cal['wealth']['log_scale'])
        res = score_paths(paths, x, ob, cal, constraints=constraint)
        if res is None:
            local = np.zeros(len(paths)); fitted_anchor = np.full(len(paths), anchor); prior_peak = np.exp(peak-anchor)
        else:
            local, fitted_anchor, prior_peak = res[:3]
        # Preserve visible DD extrema, not just overlap between two broad vertical ranges.
        ddobs = ob['dd']
        coverage = np.zeros(len(paths))
        if len(ddobs):
            grid = (ddobs.x.to_numpy()[:, None] + np.linspace(-.5, .5, 7)).ravel()
            dd = paths / np.maximum(np.maximum.accumulate(paths, axis=1), np.asarray(prior_peak).reshape(-1, 1)) - 1
            yy = (cal['dd']['y0'] - cal['dd']['scale'] * mul(dd, interpolation_matrix(x, grid))).reshape(len(paths), len(ddobs), 7)
            coverage = np.mean(np.maximum(yy.min(2) - ddobs.ymin.to_numpy() - 1, 0)**2
                + np.maximum(ddobs.ymax.to_numpy() - yy.max(2) - 1, 0)**2, axis=1) / 9
        local = local + coverage
        # Keep broad image candidates AND variance diversity among reasonable fits.
        ids = set(np.argsort(local, kind='stable')[:512].tolist())
        near = np.flatnonzero(local <= local.min() + 9)
        variance = (r * r).sum(axis=1)
        if len(near):
            order = near[np.argsort(variance[near])]
            for part in np.array_split(order, 8):
                if len(part):
                    ids.update(part[np.argsort(local[part])[:32]].tolist())
        if str(start.date()) in warm:
            for ci, cc in enumerate(combos):
                if {cols[j] for j in cc} == warm[str(start.date())]:
                    ids.add(ci); break
        ids = np.array(sorted(ids), dtype=int)
        file = folder / f'{pool}_{start.date()}.npz'
        np.savez_compressed(file, combos=combos[ids], logs=logs[ids], local=local[ids],
                            moment1=r[ids].sum(axis=1), moment2=(r[ids]**2).sum(axis=1),
                            momentq=(r[ids] * q).sum(axis=1), original_ids=ids)
        entry = dict(start=str(start.date()), end=str(end.date()), symbols=cols,
                     date_count=len(f), all_combinations=len(combos), retained_combinations=len(ids),
                     file=str(file), sha256=sha(file), prior_path=sorted(warm.get(str(start.date()), [])),
                     no_future_returns_in_eligibility_except_inverse_path_completeness=True)
        metadata.append(entry)
        if k % 12 == 0: print('CACHE', pool, k, start.date(), len(combos), len(ids), flush=True)
    dump(f'META_{pool}.json', metadata)
    dump(f'INPUT_HASHES_{pool}.json', sources)
    print('CACHE_DONE', pool, len(metadata), flush=True)


def indices_at(x, values):
    j = np.searchsorted(x, values, side='right') - 1
    j = np.clip(j, 0, len(x)-2)
    t = np.clip((values - x[j]) / (x[j+1] - x[j]), 0, 1)
    return j, t


def projected(y, idx):
    j, t = idx
    return y[:, j] * (1-t) + y[:, j+1] * t


def pack(pool):
    px, _, _ = load()
    meta = read(P / f'META_{pool}.json')
    cal = read(I / 'CALIBRATION.json'); obs = load_observations()
    dates, _ = dates_and_events(px); x = xcoord(dates, cal)
    M = len(meta); K = max(m['retained_combinations'] for m in meta); T = max(m['date_count'] for m in meta)
    L = np.zeros((M, K, T)); e = np.zeros((M, K)); one = e.copy(); two = e.copy(); cov = e.copy()
    counts, local, combos, take = [], [], [], []
    for m, entry in enumerate(meta):
        assert sha(entry['file']) == entry['sha256']
        q = np.load(entry['file']); k, t = q['logs'].shape
        L[m, :k, :t] = q['logs']; L[m, :k, t:] = q['logs'][:, -1, None]
        e[m, :k] = q['logs'][:, -1]; one[m, :k] = q['moment1']; two[m, :k] = q['moment2']; cov[m, :k] = q['momentq']
        counts.append(k); local.append(q['local']); combos.append(q['combos'])
        take.extend(m*T + np.arange(1, t))
    r = px.loc[dates, 'QQQ'].pct_change().dropna().to_numpy()
    widx, wtarget, wscale, wprecision = [], [], [], []
    for panel in ['wealth', 'relative']:
        z = obs[panel]; z = z[(z.x >= x[0]) & (z.x <= x[-1])]
        widx.extend(z.x); wtarget.extend(z.log_wealth)
        wscale.extend([cal[panel]['log_scale']]*len(z))
        wprecision.extend(1/np.maximum(1, (z.ymax-z.ymin)/2+.5)**2)
    ddobs = obs['dd']; ddobs = ddobs[(ddobs.x >= x[0]) & (ddobs.x <= x[-1])]
    grid = (ddobs.x.to_numpy()[:, None] + np.linspace(-.5, .5, 7)).ravel()
    deepest = ddobs[ddobs.ymax >= ddobs.ymax.max()-2]
    landmark = np.flatnonzero((x >= deepest.x.min()-1) & (x <= deepest.x.max()+1))
    dd_dates = START + pd.to_timedelta((ddobs.x-cal['wealth']['x0'])/cal['wealth']['width']*(END-START).days, unit='D')
    dd_month = pd.DatetimeIndex(dd_dates).to_period('M')
    dd_groups = [np.flatnonzero(dd_month == period) for period in dd_month.unique()]
    return dict(meta=meta, dates=dates, L=L, E=e, one=one, two=two, cov=cov, counts=counts,
                local=local, combos=combos, take=np.array(take), qret=r, qsum=r.sum(), q2=(r*r).sum(),
                widx=indices_at(x, np.array(widx)), wtarget=np.array(wtarget), wscale=np.array(wscale),
                wprecision=np.array(wprecision), ddidx=indices_at(x, grid), ddobs=ddobs, cal=cal,
                landmark=landmark, dd_groups=dd_groups)


def landmark_penalty(drawdown, indices):
    return ((drawdown[:, indices].min(axis=1)-TARGET['max_dd'])/.01)**2


def evaluate(choices, d, metric_weight=1.):
    choices = np.atleast_2d(choices); mi = np.arange(len(d['meta']))[None, :]
    ends = d['E'][mi, choices]; levels = np.pad(np.cumsum(ends, axis=1)[:, :-1], ((0, 0), (1, 0)))
    logpaths = d['L'][mi, choices] + levels[:, :, None]
    navlog = np.column_stack([np.zeros(len(choices)), logpaths.reshape(len(choices), -1)[:, d['take']]])
    assert navlog.shape[1] == len(d['dates'])
    dd = np.exp(navlog - np.maximum.accumulate(navlog, axis=1)) - 1
    q1 = d['one'][mi, choices].sum(1); q2 = d['two'][mi, choices].sum(1); qq = d['cov'][mi, choices].sum(1)
    n = navlog.shape[1]-1
    variance = np.maximum(1e-18, (q2-q1*q1/n)/(n-1))
    sharpe = q1/n/np.sqrt(variance)*np.sqrt(252)
    corr = (qq-q1*d['qsum']/n)/np.sqrt(np.maximum(1e-18, (q2-q1*q1/n)*(d['q2']-d['qsum']**2/n)))
    residual = (projected(navlog, d['widx'])-d['wtarget'])*d['wscale']
    image = np.mean(residual**2*d['wprecision'], axis=1)
    yy = (d['cal']['dd']['y0']-d['cal']['dd']['scale']*projected(dd, d['ddidx'])).reshape(len(choices), len(d['ddobs']), 7)
    # The observed colored stroke must be supported by a daily path within that pixel column.
    gaplo = np.maximum(yy.min(2)-d['ddobs'].ymin.to_numpy()-1, 0)
    gaphi = np.maximum(d['ddobs'].ymax.to_numpy()-yy.max(2)-1, 0)
    drawimage = np.mean(gaplo**2+gaphi**2, axis=1)/9
    maxdd = dd.min(1)
    # Optimization scales are fixed declared search priorities, NOT admission tolerances.
    metric = ((navlog[:, -1]-np.log(TARGET['wealth']))/.02)**2 + ((sharpe-TARGET['sharpe'])/.03)**2
    metric += ((corr-TARGET['correlation'])/.02)**2 + ((maxdd-TARGET['max_dd'])/.01)**2
    loss = image + drawimage + metric_weight*metric
    if d.get('draw_refinement'):
        # Rare but large month-level contradictions must not vanish in a global mean.
        column_squared = gaplo**2+gaphi**2
        month_extrema = np.mean(np.stack([column_squared[:, ids].max(axis=1) for ids in d['dd_groups']], axis=1), axis=1)/9
        metric = metric + landmark_penalty(dd, d['landmark'])
        loss = image + 2*drawimage + month_extrema + metric_weight*metric
    if 'factor_penalty' in d:
        loss = loss + 20*np.mean(d['factor_penalty'][mi, choices], axis=1)
    return loss, dict(image=image, draw_image=drawimage, metric=metric,
        wealth=np.exp(navlog[:, -1]), cagr=np.exp(navlog[:, -1]*252/n)-1,
        sharpe=sharpe, max_dd=maxdd, correlation=corr, volatility=np.sqrt(variance*252))


def path_series(choice, d):
    mi = np.arange(len(choice)); ends = d['E'][mi, choice]
    levels = np.r_[0., np.cumsum(ends)[:-1]]
    p = d['L'][mi, choice] + levels[:, None]
    return pd.Series(np.exp(np.r_[0., p.ravel()[d['take']]]), index=d['dates'])


def run(pool, seed_mode, rounds=5, refine=False):
    d = pack(pool); M = len(d['meta'])
    choice = np.array([int(np.argmin(q)) for q in d['local']])
    if seed_mode == 'prior':
        for m, z in enumerate(d['meta']):
            if not z['prior_path']: continue
            for k, cc in enumerate(d['combos'][m]):
                if {z['symbols'][j] for j in cc} == set(z['prior_path']):
                    choice[m] = k; break
    ident = f'{pool}__{seed_mode}'
    if refine:
        choice = np.load(P/'internal_states'/f'{ident}_states.npz')['final']
        ident += '__draw_v2'
        d['draw_refinement'] = True
    folder = P / 'internal_states'; folder.mkdir(exist_ok=True)
    trial_files = []; trace = []; state_archive = [choice.copy()]
    initial_loss, initial_details = evaluate(choice, d)
    initial = {k: float(v[0]) for k, v in initial_details.items()}
    rng = np.random.default_rng(20261009)
    for iteration in range(rounds):
        evaluations = []; changes = 0
        # First attend to the visible largest DD; then visit every month.
        seq = rng.permutation(M)
        if iteration == 0:
            critical = [m for m, z in enumerate(d['meta']) if z['start'].startswith(('2026-06', '2026-07', '2020-02', '2025-03'))]
            seq = np.r_[critical, [m for m in seq if m not in critical]]
        weight = 4. if refine or iteration >= 3 else 1.
        for counter, m in enumerate(seq):
            candidates = np.repeat(choice[None, :], d['counts'][m], axis=0)
            candidates[:, m] = np.arange(d['counts'][m])
            losses = []
            for b in range(0, len(candidates), 64):
                losses.extend(evaluate(candidates[b:b+64], d, weight)[0])
            losses = np.array(losses); best = int(np.argmin(losses))
            evaluations.append(dict(month=int(m), before=int(choice[m]), after=best, losses=losses.tolist()))
            if best != choice[m]: changes += 1
            choice[m] = best; state_archive.append(choice.copy())
            if counter % 30 == 0: print('SEARCH', ident, iteration, counter, 'loss', round(float(losses[best]), 4), flush=True)
        loss, metrics = evaluate(choice, d, weight)
        item = dict(iteration=iteration, metric_weight=weight, changes=changes, loss=float(loss[0]), **{k: float(v[0]) for k, v in metrics.items()})
        trace.append(item)
        trial = folder / f'{ident}_{iteration}.json'; trial.write_text(json.dumps(evaluations, separators=(',', ':')))
        trial_files.append(dict(file=str(trial), sha256=sha(trial)))
        np.savez_compressed(folder / f'{ident}_states.npz', states=np.array(state_archive), final=choice)
        dump(f'{ident}_PROGRESS.json', dict(initial=initial, trace=trace, trial_files=trial_files))
        print('ROUND', ident, json.dumps(item), flush=True)
        if changes == 0 and iteration >= 3: break
    nav = path_series(choice, d)
    nav.to_csv(P / f'{ident}_NAV.csv')
    # No public transaction list. Store only non-admitted latent choice IDs for reproducibility.
    record = dict(pool=pool, initialization=seed_mode, **trace[-1], initial=initial, trace=trace,
                  exact_window=[str(nav.index[0].date()), str(nav.index[-1].date())],
                  future_prices_used_for_inverse_attribution=True, author_strategy_reproduced=False,
                  candidate_transactions_exported=False, uniform_selection_rule_found=False,
                  metric_targets_read_from_supplied_figure=TARGET)
    dump(f'{ident}_RESULT.json', record)
    print('SEARCH_DONE', ident, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('phase', choices=['prepare', 'search']); ap.add_argument('--pool', choices=['core17', 'theme17'], required=True)
    ap.add_argument('--seed', choices=['prior', 'local'], default='prior'); ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--refine', action='store_true')
    args = ap.parse_args()
    if args.phase == 'prepare': prepare(args.pool)
    else: run(args.pool, args.seed, args.rounds, args.refine)


if __name__ == '__main__':
    main()
