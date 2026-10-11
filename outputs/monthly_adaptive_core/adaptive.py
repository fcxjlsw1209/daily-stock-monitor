"""Adaptive long-momentum reconstruction. Never modifies observation accounts."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import itertools
import sys
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
O = P.parent
sys.path.insert(0, str(O / 'monthly_style_inverse'))
import style_study as old
e, policy, strict = old.e, old.policy, old.strict
read, sha = old.read, old.sha
FEATURE_MODES = ['legacy', 'risk_path']
CORE_MENUS = dict(
    fixed=['calendar14_skip2_half'],
    durations=['mom252_skip42', 'mom294_skip42', 'calendar12_skip2_raw', 'calendar14_skip2_raw'],
    risk_methods=['mom252_risk_recent63', 'mom294_risk_recent63', 'mom252_risk_aligned', 'mom294_risk_aligned',
                  'calendar12_skip2_half', 'calendar14_skip2_half', 'calendar12_skip2_risk', 'calendar14_skip2_risk'],
    broad=['mom252_skip42', 'mom294_skip42', 'calendar12_skip2_raw', 'calendar14_skip2_raw',
           'mom252_risk_recent63', 'mom294_risk_recent63', 'mom252_risk_aligned', 'mom294_risk_aligned',
           'calendar12_skip2_half', 'calendar14_skip2_half', 'calendar12_skip2_risk', 'calendar14_skip2_risk',
           'trend_quality252_skip42', 'trend_quality294_skip42', 'mean_sharpe252_skip42', 'mean_sharpe294_skip42'])
FOLDS = [(17, 23), (23, 35)]


def dump(name, value):
    (P / name).write_text(old.json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda v: v.item() if isinstance(v, np.generic) else v) + '\n')


def design(d, core, mode):
    spec = [(k, 1., 'core') for k in core]
    volnames = ['vol126'] + (['downside63', 'idio63', 'vol_ratio21_126'] if mode == 'risk_path' else [])
    spec += [(k, sign, 'vol') for k in volnames for sign in [1., -1.]]
    spec += [(f'stock_sector_skip42_{n}', 1., 'industry') for n in [126, 252]]
    shapes = ['cmf21', 'cmf63', 'signed_volume63', 'volume_ratio21_126', 'highvol_cmf63', 'trend_cmf63']
    if mode == 'risk_path':
        shapes += ['high252_distance', 'close_location21', 'jump_concentration63']
    spec += [(k, sign, 'shape') for k in shapes for sign in [1., -1.]]
    spec += [('acceleration_skip42', 1., 'accel')]
    X = np.stack([d['X'][:, :, d['names'].index(k)] * sign for k, sign, group in spec], axis=-1)
    return X, spec


def initialize():
    assert not (P / 'PLAN.json').exists()
    dump('PLAN.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        acceptance='All69 monthly author three-name sets and ordered lists, monthend changes, and rounded monthly-return replay. Do not count lookup tables or per-month oracle fits as reproduction.',
        task='Extend long-core mixtures so observable market regimes can change momentum horizon/risk normalization; add bounded stock-specific risk/price-path corrections.',
        unchanged='Only monthend decisions, fixed adjusted units within month, zero cost. Long/industry returns exclude42sessions/two calendar months. No daily pullback entry gate, no short-horizon return factor, no author state fed into portfolio.',
        pool='Separate current research Top32 and previously user-approved Top35 controls, same theme17. No one-off COST or author-name additions; 2022-05-31 COST is proxy rank36, ~0.103% below rank35.',
        core_menus=CORE_MENUS, feature_modes=FEATURE_MODES,
        path_definition='Downside63, idio63, volatility21/126 use recent risk; distance to252close-high, mean intraday close-location21, squared-return jump concentration63 are bounded price-volume/path corrections, never core momentum.',
        weights='Same constraints: core>=50%, vol<=30%, own-sector relative<=20%, all shape/accel<=20%, accel<=10%; these remain research assumptions, not disclosed author weights.',
        contexts='Global, growth(QQQ/SPY63), smallcap(IWM/SPY63), defensive-minus-cyclical63, bonds(TLT63), growth x high SPYvol. Group requires8 calibration months; otherwise global. State L1 weight delta<=.30.',
        fit='Constrained ordered likelihood temp.05, regularization.10. 4core menus x2feature modes x2norms x2pools x6contexts x4retention=768candidate policies. Two forward inner folds: fit17/eval6, fit23/eval12; first35 only.',
        select='Aggregate inner18months: .60 slot recall+.25 exactset+.15 exactorder. Prefer simpler policy on ties. Freeze matching leader, plus adaptive and risk/path category leaders, then refit selected specification on35.',
        validation='Outer34 unchanged (2023-11-30 to2026-09-24). Previously reused history is not pristine OOS. Do not use outer labels or returns to select candidates.',
        diagnostics='After frozen validation, optionally all35 per-month feasibility identifies whether added features explain training contradictions. No oracle-generated trade performance.',
        protected='Read-only original strategy accounts, ledgers, plans, fingerprints, input datasets and prior research.'))
    prev = read(O / 'monthly_style_inverse/PROTECTED_BEFORE.json')
    dump('PROTECTED_BEFORE.json', {f: sha(f) for f in prev})


def prepare():
    px, raw, members, periods, sources, pools = e.load_prices()
    feat, _, states = e.previous.panels(px, raw)
    core, side = e.features(px, raw)
    names = list(dict.fromkeys(k for menu in CORE_MENUS.values() for k in menu))
    clean = {k: feat[k] if k in feat else core[k] for k in names}
    side_names = ['vol126', 'cmf21', 'cmf63', 'signed_volume63', 'volume_ratio21_126', 'acceleration_skip42',
                  'downside63', 'idio63', 'vol_ratio21_126', 'high252_distance', 'close_location21', 'jump_concentration63']
    clean.update({k: side[k] for k in side_names})
    rel = {f'stock_sector_skip42_{n}': px.shift(42) / px.shift(n) - 1 for n in [126, 252]}
    names = list(clean) + list(rel) + ['highvol_cmf63', 'trend_cmf63']
    sf = O / 'index_union/alias_resolved_inputs/sectors.csv.gz'
    sectors = pd.read_csv(sf, index_col=0, parse_dates=True, low_memory=False)
    sources[str(sf)] = sha(sf)
    warm = px.notna().rolling(295).sum().eq(295) & px.gt(0)
    mom = px.shift(42) / px.shift(252) - 1
    metadata = dict(periods=periods, packs={}, source_limits='Reconstructed cap pool, proxy industry history, assumed ETF list; not original vendor/model data.')
    for pool in [32, 35]:
        symbols = [sorted({e.actual(x) for x in pools[str(pool)][p['start']]} | members['theme17'][1]) for p in periods]
        for norm in ['percentile', 'magnitude']:
            key = f'top{pool}_{norm}'
            X = np.zeros((len(periods), max(map(len, symbols)), len(names)))
            eligible = np.zeros(X.shape[:2], bool)
            labels = np.zeros_like(eligible)
            order = np.full((len(periods), 3), -1, int)
            coverage = []
            for i, (p, ss) in enumerate(zip(periods, symbols)):
                date = p['start']
                bm = strict.benchmark_map(date, ss, sectors, members['theme17'][1])
                tab = pd.DataFrame({k: z.loc[date, ss] for k, z in clean.items()})
                for k, z in rel.items():
                    tab[k] = strict.relative_row(z, date, ss, bm)
                q = e.engine.normalize(tab, norm)
                q['highvol_cmf63'] = q.cmf63 * states.at[date, 'high_market_vol']
                q['trend_cmf63'] = q.cmf63 * states.at[date, 'spy_trend']
                X[i, :len(ss)] = q[names]
                eligible[i, :len(ss)] = (warm.loc[date, ss] & mom.loc[date, ss].gt(0)).to_numpy()
                for j, name in enumerate(p['holdings']):
                    symbol = e.actual(name)
                    if symbol in ss:
                        k = ss.index(symbol)
                        labels[i, k] = True
                        if eligible[i, k]:
                            order[i, j] = k
                coverage.append(dict(date=date, author_unavailable=[a for j, a in enumerate(p['holdings']) if order[i, j] < 0]))
            np.savez_compressed(P / (key + '.npz'), X=X, eligible=eligible, labels=labels, order=order)
            metadata['packs'][key] = dict(symbols=symbols, names=names, norm=norm, pool=pool, coverage=coverage)
    for mod in [Path(__file__), Path(old.__file__), Path(strict.__file__), Path(policy.__file__), Path(e.__file__)]:
        sources[str(mod)] = sha(mod)
    dump('INPUTS.json', metadata)
    dump('SOURCE_HASHES.json', sources)
    print('Prepared four point-in-time packs, feature count', len(names), flush=True)


def load():
    m = read(P / 'INPUTS.json')
    packs = {k: dict(np.load(P / (k + '.npz')), **v) for k, v in m['packs'].items()}
    px, raw, members, periods, sources, pools = e.load_prices()
    opening = raw['opening'] * px / raw['close']
    dates = pd.to_datetime([p['start'] for p in periods])
    symbols = sorted({s for d in packs['top35_percentile']['symbols'] for s in d})
    ids = {s: j for j, s in enumerate(symbols)}
    styles = old.market_styles(px, dates)
    sigma = px.pct_change(fill_method=None).rolling(63).std() * np.sqrt(21)
    peaks = []
    for i, date in enumerate(dates):
        start = date if not i else px.index[px.index.get_loc(dates[i - 1]) + 1]
        peaks.append(px.loc[start:date, symbols].max().to_numpy())
    opdates = [px.index[px.index.get_loc(d) + 1] for d in dates]
    context = dict(symbols=symbols, ids=ids, dates=[str(d.date()) for d in dates],
        close=px.loc[dates, symbols].to_numpy(), opening=opening.loc[opdates, symbols].to_numpy(),
        opening_dates=[str(d.date()) for d in opdates], peak=np.stack(peaks), sigma=sigma.loc[dates, symbols].to_numpy(),
        trend=styles.spy_above200.astype(int).to_numpy(), highvol=styles.highvol.astype(int).to_numpy(), periods=periods)
    return m, packs, px, opening, context, styles


def context_for(d, c):
    answer = dict(c)
    answer['eligible'] = np.zeros((len(c['dates']), len(c['symbols'])), bool)
    for i, ss in enumerate(d['symbols']):
        answer['eligible'][i, [c['ids'][s] for s in ss]] = d['eligible'][i, :len(ss)]
    return answer


def fit(X, spec, d, groups, mask, base=None):
    if base is None:
        base, _ = e.fit(X, d, mask, spec, .05, .10)
    weights, audit = {}, {}
    for g in sorted(set(groups)):
        if len(set(groups)) == 1:
            w, a = base.copy(), dict(months=int(mask.sum()), l1_delta=0.)
        else:
            w, a = policy.fit_delta(X, d, mask & (groups == g), spec, base, .15, .05)
        weights[str(g)], audit[str(g)] = w.tolist(), a
    return dict(spec=spec, base=base.tolist(), weights=weights, audit=audit)


def scores_for(rule, X, spec, d, c, groups):
    W = np.array([rule['weights'][str(g)] for g in groups])
    local = np.einsum('ijk,ik->ij', X, W, optimize=False)
    scores = np.full(c['eligible'].shape, -np.inf)
    strength = np.zeros_like(scores)
    ci = [j for j, t in enumerate(spec) if t[2] == 'core']
    for i, ss in enumerate(d['symbols']):
        js = [c['ids'][s] for s in ss]
        scores[i, js] = local[i, :len(ss)]
        q = X[i, :len(ss)][:, ci].mean(axis=1)
        strength[i, js] = pd.Series(q).where(d['eligible'][i, :len(ss)]).rank(pct=True).fillna(.5)
    return scores, strength, local, W


def choose(row):
    mm = row['inner_validation']
    metric = .6 * mm['shared_slots'] / (3 * mm['months']) + .25 * mm['set_exact'] / mm['months'] + .15 * mm['order_exact'] / mm['months']
    return metric, -(row['partition'] != 'global'), -(row['mode'] != 'legacy'), -row['config']['buffer'], -len(CORE_MENUS[row['menu']]), -row['id']


def train():
    m, packs, px, opening, basecontext, styles = load()
    rows = []
    for key, menu, mode in itertools.product(packs, CORE_MENUS, FEATURE_MODES):
        d = packs[key]
        c = context_for(d, basecontext)
        X, spec = design(d, CORE_MENUS[menu], mode)
        aggregations = {(part, b, gap): [] for part in old.PARTITIONS for b, gap in old.RETENTION}
        for before, until in FOLDS:
            training = np.arange(69) < before
            val = (np.arange(69) >= before) & (np.arange(69) < until)
            base, _ = e.fit(X, d, training, spec, .05, .10)
            for part in old.PARTITIONS:
                groups = styles[part].to_numpy()
                rule = fit(X, spec, d, groups, training, base)
                scores, strength, local, W = scores_for(rule, X, spec, d, c, groups)
                for b, gap in old.RETENTION:
                    cfg = dict(buffer=b, gap=gap, exit=dict(name='none'))
                    sim = policy.simulate(c, scores, strength, cfg, count=until)
                    metric = policy.matching(sim['choices'], scores, c, val)
                    aggregations[(part, b, gap)].append(metric)
        for (part, b, gap), mm in aggregations.items():
            summed = {k: sum(z[k] for z in mm) for k in ['months', 'shared_slots', 'set_exact', 'order_exact']}
            rows.append(dict(id=len(rows), key=key, pool=d['pool'], menu=menu, mode=mode, partition=part,
                             config=dict(buffer=b, gap=gap, exit=dict(name='none')), inner_validation=summed, inner_folds=mm))
        print('Candidate policies finished', len(rows), key, menu, mode, flush=True)
    assert len(rows) == 768
    selected = {name: max(group, key=choose) for name, group in [
        ('overall', rows), ('top32', [r for r in rows if r['pool'] == 32]),
        ('adaptive', [r for r in rows if r['menu'] != 'fixed' and r['partition'] != 'global']),
        ('risk_path', [r for r in rows if r['mode'] == 'risk_path'])]}
    frozen = {}
    for name, row in selected.items():
        d = packs[row['key']]
        X, spec = design(d, CORE_MENUS[row['menu']], row['mode'])
        rule = fit(X, spec, d, styles[row['partition']].to_numpy(), np.arange(69) < 35)
        frozen[name] = dict(**row, rule=rule)
    dump('INNER_TRIALS.json', rows)
    dump('FROZEN.json', dict(created_at=datetime.now(timezone.utc).isoformat(), leaders=frozen,
                             training_months=35, validation_months=34, outer_returns_not_used=True))
    print('FROZEN', {n: {k: r[k] for k in ['key', 'menu', 'mode', 'partition', 'config', 'inner_validation']} for n, r in frozen.items()}, flush=True)


def evaluate():
    h = sha(P / 'FROZEN.json')
    assert all(sha(f) == v for f, v in read(P / 'SOURCE_HASHES.json').items())
    m, packs, px, opening, basecontext, styles = load()
    ms = old.masks(69)
    results = {}
    for name, row in read(P / 'FROZEN.json')['leaders'].items():
        d = packs[row['key']]
        c = context_for(d, basecontext)
        X, spec = design(d, CORE_MENUS[row['menu']], row['mode'])
        scores, strength, local, W = scores_for(row['rule'], X, spec, d, c, styles[row['partition']].to_numpy())
        sim = policy.simulate(c, scores, strength, row['config'], detailed=True)
        met, slices, eq, trades = old.performance(px, opening, c, sim, scores, 'close')
        os = policy.simulate(c, scores, strength, row['config'], execution='open')
        om, oss, _, _ = old.performance(px, opening, c, os, scores, 'open')
        matching = {k: policy.matching(sim['choices'], scores, c, ms[k]) for k in ['training', 'validation', 'all']}
        results[name] = dict(metrics=met, slices=slices, next_open=om, next_open_slices=oss, matching=matching, rule=row)
        dump(name + '_TRADES.json', trades)
        dump(name + '_DECISIONS.json', sim['events'])
        dump(name + '_DAILY.json', [dict(date=str(t.date()), equity=float(v)) for t, v in eq.items()])
        if name == 'overall':
            months, txt = old.explain(row, d, c, styles, scores, local, X, W, sim)
            dump('MONTHLY_EXPLANATIONS.json', months)
            (P / 'MONTHLY_EXPLANATIONS.txt').write_text(txt)
        print(name, dict(cagr=met['cagr'], sharpe=met['sharpe'], validation=slices['validation34'], matching=matching), flush=True)
    oldr = read(O / 'monthly_style_inverse/RESULTS.json')
    for name in ['author', 'qqq']:
        results[name] = oldr[name]
    dump('RESULTS.json', results)
    assert sha(P / 'FROZEN.json') == h
    changes = [f for f, v in read(P / 'PROTECTED_BEFORE.json').items() if sha(f) != v]
    assert not changes, changes
    dump('VALIDATION.json', dict(protected_changes=changes, frozen_hash=h, source_hashes_verified=True,
        inner_forward_folds=FOLDS, train35_validate34=True, no_outer_labels_or_returns_used_for_selection=True,
        no_intramonth_trades=True, zero_cost=True, independent_share_ledger_replay=True, not_pristine_oos=True))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['initialize', 'prepare', 'train', 'evaluate'])
    globals()[ap.parse_args().phase]()
