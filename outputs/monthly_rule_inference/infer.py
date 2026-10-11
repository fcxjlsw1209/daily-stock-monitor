"""Conditional inverse-policy inference. Labels are hindsight hypotheses.

Fits do not access future returns; evaluation uses subsequent prices. Label
construction itself used the entire reference image, so year holdouts test
explanatory stability, NOT independent investment performance.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
O = P.parent
sys.path.insert(0, str(O / 'monthly_path_constraints'))
from study import load, sha, read, START, END

CORE = ['mom12_2', 'mom9_2', 'mom6_2', 'risk12_2', 'risk9_2', 'risk6_2']
SIDE = ['vol21', 'vol63', 'vol126', 'downside63', 'industry_relative60',
        'cmf21', 'cmf63', 'volume_ratio21_126', 'near_high252', 'ma126_distance']
INTER = ['mom12_vol63', 'mom6_vol63', 'mom12_industry', 'mom12_high', 'cmf21_vol63', 'mom12_market_trend']
MODELS = ['raw_momentum', 'core50_side50', 'core60_side40', 'core80_side20', 'core60_bounded_ridge', 'unconstrained_ridge_diagnostic']
ALPHA = .001
MARGIN = .10


def dump(name, value):
    (P / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def feature_frames(px, raw):
    ret = px.pct_change(fill_method=None)
    vols = {n: ret.rolling(n).std() for n in [21, 63, 126]}
    mom = {n: px.shift(42) / px.shift(n) - 1 for n in [126, 189, 252]}
    c, h, l, v = [raw[k] for k in ['close', 'high', 'low', 'volume']]
    flow = ((2*c-h-l)/(h-l).replace(0, np.nan)).where((h-l).ne(0), 0)
    f = {f'mom{n // 21}_2': m for n, m in mom.items()}
    f.update({f'risk{n // 21}_2': m / vols[63].replace(0, np.nan) for n, m in mom.items()})
    f.update({f'vol{n}': value for n, value in vols.items()})
    f.update(downside63=np.minimum(ret, 0).pow(2).rolling(63).mean().pow(.5),
        industry_relative60=(1 + px.pct_change(60, fill_method=None))/(1 + raw['sector60']) - 1,
        cmf21=(flow*v).rolling(21).sum()/v.rolling(21).sum(),
        cmf63=(flow*v).rolling(63).sum()/v.rolling(63).sum(),
        volume_ratio21_126=v.rolling(21).mean()/v.rolling(126).mean(),
        near_high252=px/px.rolling(252).max(), ma126_distance=px/px.rolling(126).mean()-1)
    return f


def prepare():
    px, pools, sources = load()
    records_file = O / '01a0b6b1-511d-7401-b4f7-4dd2aaa895a7/monthly_inverse_holdings/records.json'
    records = read(records_file)
    sources.update(records['sources'])
    sources[str(records_file)] = sha(records_file)
    raw = {}
    for field in ['close', 'high', 'low', 'volume', 'sector60']:
        file = O / 'index_union/alias_resolved_inputs' / f'{field}.csv.gz'
        raw[field] = pd.read_csv(file, index_col=0, parse_dates=True)
        sources[str(file)] = sha(file)
    manifests = read(O / 'four_factor_rebound/ETF_SOURCES.json') + read(O / 'monthly_reconstruction_v2/ETF_MANIFEST.json')
    etfs = {r['symbol'] for r in manifests}
    for entry in manifests:
        z = pd.read_csv(entry['file'], index_col=0, parse_dates=True)
        for field, col in [('close', 'Close'), ('high', 'High'), ('low', 'Low'), ('volume', 'Volume')]:
            raw[field][entry['symbol']] = z[col].reindex(px.index)
        raw['sector60'][entry['symbol']] = px.SPY.pct_change(60, fill_method=None)
    raw = {k: f.reindex(index=px.index, columns=px.columns) for k, f in raw.items()}
    feat = feature_frames(px, raw)
    member = pools['theme17']
    ranked = {k: f.where(member).rank(axis=1, pct=True).fillna(.5) - .5 for k, f in feat.items()}
    good = member & feat['mom12_2'].gt(0) & px.notna().rolling(253, min_periods=253).sum().eq(253) & px.gt(0)
    market_regime = np.sign(px.SPY / px.SPY.rolling(200).mean() - 1)
    rows, diagnostics = [], []
    for event in records['monthly']:
        date = pd.Timestamp(event['rebalance_date'])
        symbols = sorted(px.columns[good.loc[date]])
        labels = np.asarray([s in event['source_holdings'] for s in symbols])
        assert labels.sum() == 3
        X = np.column_stack([ranked[k].loc[date, symbols] for k in CORE + SIDE])
        # Core remains positive; all scores use percentile inputs in [-0.5,0.5].
        assert np.isfinite(X).all() and np.abs(X).max() <= .5 + 1e-12
        col = {k: X[:, i] for i, k in enumerate(CORE + SIDE)}
        Z = np.column_stack([col['mom12_2']*col['vol63'], col['mom6_2']*col['vol63'],
            col['mom12_2']*col['industry_relative60'], col['mom12_2']*col['near_high252'],
            col['cmf21']*col['vol63'], col['mom12_2']*float(market_regime.loc[date])])
        rows.append(dict(date=date, end=pd.Timestamp(event['through_date']), symbols=symbols, X=X, Z=Z, y=labels))
        for i, s in enumerate(symbols):
            diagnostics.append(dict(date=str(date.date()), symbol=s, selected=bool(labels[i]), is_etf=s in etfs,
                **{k: float(X[i, j] + .5) for j, k in enumerate(CORE + SIDE)},
                missing_features=[k for k in CORE + SIDE if pd.isna(feat[k].at[date, s])]))
    # Prefix recomputation catches accidental future-dependent feature construction.
    for cut in ['2019-12-31', '2022-12-30', '2025-12-31']:
        limited = feature_frames(px.loc[:cut], {k: z.loc[:cut] for k, z in raw.items()})
        for k in feat:
            assert np.allclose(limited[k].iloc[-1], feat[k].loc[cut], equal_nan=True, rtol=1e-12, atol=1e-12), (cut, k)
    return px, rows, diagnostics, sources


def simplex(x, radius):
    u = np.sort(x)[::-1]
    css = np.cumsum(u)
    rho = np.flatnonzero(u - (css-radius)/np.arange(1, len(u)+1) > 0)[-1]
    return np.maximum(x - (css[rho]-radius)/(rho+1), 0)


def l1ball(x, radius):
    return x if np.abs(x).sum() <= radius else np.sign(x)*simplex(np.abs(x), radius)


def pair_data(rows):
    ds, zs, weights = [], [], []
    for row in rows:
        yes, no = np.flatnonzero(row['y']), np.flatnonzero(~row['y'])
        d = (row['X'][yes, None] - row['X'][no][None, :]).reshape(-1, len(CORE+SIDE))
        z = (row['Z'][yes, None] - row['Z'][no][None, :]).reshape(-1, len(INTER))
        ds.append(d); zs.append(z); weights.extend([1/len(d)/len(rows)]*len(d))
    return np.vstack(ds), np.vstack(zs), np.asarray(weights)


def constrained_ridge(D, weights, mass):
    gram = np.einsum('ni,n,nj->ij', D, weights, D, optimize=False) + ALPHA*np.eye(D.shape[1])
    rhs = MARGIN*np.einsum('ni,n->i', D, weights, optimize=False)
    L = float(np.linalg.eigvalsh(gram)[-1])
    w = np.r_[np.full(len(CORE), mass/len(CORE)), np.zeros(len(SIDE))]
    accelerated = w.copy(); t = 1.
    for iteration in range(1, 6001):
        step = accelerated - (gram@accelerated-rhs)/L
        candidate = np.r_[simplex(step[:len(CORE)], mass), l1ball(step[len(CORE):], 1-mass)]
        if np.linalg.norm(candidate-w) < 1e-10:
            w = candidate; break
        nt = (1+np.sqrt(1+4*t*t))/2
        accelerated = candidate + (t-1)/nt*(candidate-w)
        w, t = candidate, nt
    assert np.all(w[:len(CORE)] >= -1e-12)
    assert abs(w[:len(CORE)].sum()-mass) < 1e-10
    assert np.abs(w[len(CORE):]).sum() <= 1-mass+1e-10
    return w, iteration


def fit(rows, name):
    D, Z, weights = pair_data(rows)
    if name == 'raw_momentum':
        w = np.zeros(D.shape[1]); w[0] = .8; w[2] = .2
        return dict(name=name, w=w, iterations=0)
    if name == 'unconstrained_ridge_diagnostic':
        gram = np.einsum('ni,n,nj->ij', D, weights, D, optimize=False) + ALPHA*np.eye(D.shape[1])
        rhs = MARGIN*np.einsum('ni,n->i', D, weights, optimize=False)
        return dict(name=name, w=np.linalg.solve(gram, rhs), iterations=1)
    mass = .5 if name.startswith('core50') else (.8 if name.startswith('core80') else .6)
    w, it = constrained_ridge(D, weights, mass)
    result = dict(name=name, w=w, iterations=it)
    if name == 'core60_bounded_ridge':
        residual = MARGIN - .9*np.einsum('ni,i->n', D, w, optimize=False)
        gram = np.einsum('ni,n,nj->ij', Z, weights, Z, optimize=False) + ALPHA*np.eye(Z.shape[1])
        rhs = np.einsum('ni,n,n->i', Z, weights, residual, optimize=False)
        result['z'] = np.linalg.solve(gram, rhs)
    return result


def score(model, row):
    result = np.einsum('ni,i->n', row['X'], model['w'], optimize=False)
    if 'z' in model:
        correction = np.clip(np.einsum('ni,i->n', row['Z'], model['z'], optimize=False), -.05, .05)
        assert np.abs(correction).max() <= .05
        result = .9*result + correction
    return result


def evaluate(rows, scores, px):
    navs, comparisons = [], []
    capital = 1.
    pair_correct = []
    for row, values in zip(rows, scores):
        chosen = np.argsort(-values, kind='stable')[:3]
        actual = set(np.flatnonzero(row['y']))
        orders = [row['symbols'][j] for j in chosen]
        prices = px.loc[row['date']:row['end'], orders]
        assert prices.notna().all().all() and prices.gt(0).all().all(), (row['date'], orders)
        segment = capital*(prices/prices.iloc[0]).mean(axis=1)
        capital = float(segment.iloc[-1]); navs.append(segment)
        ranks = pd.Series(values).rank(ascending=False, method='min').to_numpy()
        yes, no = values[row['y']], values[~row['y']]
        gaps = yes[:, None] - no[None, :]
        accuracy = float(((gaps>0)+.5*(gaps==0)).mean()); pair_correct.append(accuracy)
        comparisons.append(dict(date=str(row['date'].date()), inferred_holdings=orders,
            target_holdings=[row['symbols'][j] for j in sorted(actual)],
            overlap=len(actual & set(chosen)), exact=actual == set(chosen),
            target_ranks=sorted(float(ranks[j]) for j in actual), candidate_count=len(values),
            minimum_arbitrary_per_security_correction=max(0.,float(no.max()-yes.min()))/2,
            pairwise_accuracy=accuracy))
    nav = pd.concat(navs); nav = nav[~nav.index.duplicated(keep='last')]
    r = nav.pct_change().dropna()
    summary = dict(months=len(rows), exact_months=sum(v['exact'] for v in comparisons),
        average_overlap=float(np.mean([v['overlap'] for v in comparisons])),
        selected_name_recall=float(np.mean([v['overlap']/3 for v in comparisons])),
        pairwise_accuracy=float(np.mean(pair_correct)),
        median_target_rank=float(np.median([rank for v in comparisons for rank in v['target_ranks']])),
        months_needing_arbitrary_correction_over_005=sum(v['minimum_arbitrary_per_security_correction']>.05 for v in comparisons),
        cagr_nav_count=float(nav.iloc[-1]**(252/len(nav))-1),
        sharpe=float(r.mean()/r.std()*np.sqrt(252)),
        max_dd=float((nav/nav.cummax()-1).min()), wealth=float(nav.iloc[-1]),
        start=str(nav.index[0].date()), end=str(nav.index[-1].date()))
    return summary, comparisons, nav


def serial(model):
    return dict(model=model['name'], weights=dict(zip(CORE+SIDE, model['w'].tolist())),
        interaction_weights=dict(zip(INTER, model['z'].tolist())) if 'z' in model else {},
        iterations=model['iterations'])


def main():
    px, rows, diagnostics, sources = prepare()
    dump('SOURCE_HASHES.json', sources)
    dump('FEATURE_DIAGNOSTICS.json', diagnostics)
    dd = pd.DataFrame(diagnostics)
    features = {}
    for name in CORE+SIDE:
        yes, no = dd.loc[dd.selected, name], dd.loc[~dd.selected, name]
        features[name] = dict(selected_median=float(yes.median()), selected_mean=float(yes.mean()),
            unselected_median=float(no.median()), selected_share_top_quartile=float((yes>=.75).mean()),
            selected_share_below_median=float((yes<.5).mean()))
    prevalence = dict(features=features, selected_names=int(dd.selected.sum()),
        selected_etf_share=float(dd.loc[dd.selected, 'is_etf'].mean()),
        candidate_etf_share=float(dd.is_etf.mean()), missing_feature_records=int(dd.missing_features.str.len().gt(0).sum()),
        caution='All labels already pass positive R12-2 and warmup by construction; that eligibility is not independently discovered.')
    dump('PREVALENCE.json', prevalence)
    results, all_curves, all_models = {}, {}, {}
    for name in MODELS:
        model = fit(rows, name)
        summary, comparison, nav = evaluate(rows, [score(model, r) for r in rows], px)
        dump(f'{name}_FULL_EXPLANATION.json', comparison)
        all_curves[name] = nav
        all_models[name] = serial(model)
        hold_rows, hold_scores, folds = [], [], []
        for year in range(2020, 2027):
            train = [r for r in rows if r['date'].year < year]
            test = [r for r in rows if r['date'].year == year]
            fm = fit(train, name)
            assert max(r['date'] for r in train) < min(r['date'] for r in test)
            scores = [score(fm, row) for row in test]
            fold, _, _ = evaluate(test, scores, px)
            folds.append(dict(test_year=year, train_through=str(train[-1]['date'].date()),
                training_months=len(train), **serial(fm), **fold))
            hold_rows.extend(test); hold_scores.extend(scores)
        hold, hold_comparison, hold_nav = evaluate(hold_rows, hold_scores, px)
        dump(f'{name}_YEAR_HOLDOUT.json', hold_comparison)
        results[name] = dict(full_fit=summary, expanding_year_holdout=hold, folds=folds,
            eligible_author_hypothesis=not name.endswith('_diagnostic'),
            genuine_investment_out_of_sample=False)
        all_curves[name+'__year_holdout'] = hold_nav
        print(name, json.dumps(dict(full=summary, holdout=hold)), flush=True)
        dump('RESULTS.json', results)
    dump('FITTED_MODELS.json', all_models)
    pd.DataFrame(all_curves).to_csv(P / 'NAV.csv.gz')
    # Selection is explanatory and independent of CAGR/Sharpe, with no refit after
    # inspecting the holdout. Disclose that this comparison itself is retrospective.
    allowed = [k for k in results if results[k]['eligible_author_hypothesis']]
    best = max(allowed, key=lambda k: (results[k]['expanding_year_holdout']['average_overlap'],
                                    results[k]['expanding_year_holdout']['pairwise_accuracy']))
    dump('SELECTION.json', dict(model=best, criterion='expanding-year target overlap then pairwise accuracy; retrospective comparison',
        rule_recovered=results[best]['full_fit']['exact_months']==len(rows),
        author_trades_independently_verified=False, genuine_investment_out_of_sample=False))
    for path, value in sources.items():
        assert sha(path) == value, path
    dump('VALIDATION.json', dict(status='passed', source_files_unchanged=len(sources),
        decision_feature_prefix_checks=3*len(CORE+SIDE), model_fits=len(MODELS)*8,
        positive_momentum_simplex_checked=True, signed_side_budget_checked=True,
        small_model_bound_checked=True, year_training_cutoffs_checked=True,
        original_accounts_unchanged=True, original_research_unchanged=True,
        target_labels_changed=False, actual_author_rule_recovered=False))
    dump('CODE_HASHES.json', {str(f):sha(f) for f in P.glob('*.py')})
    print('DONE', best, flush=True)


if __name__ == '__main__':
    main()
