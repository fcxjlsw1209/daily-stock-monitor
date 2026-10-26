"""Predeclared conditional monthly models; research only, no live changes.

This driver owns registration, causal weak-state construction, fixed support
corrections and frozen selection/replay. Model fitting lives in sibling modules.
A 70% objective is not a return promise. History has prior research exposure.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib.util
import itertools
import json
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parent
O = P.parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


previous = module('conditional70_previous', O/'monthly_target70_research/study.py')
old = previous.old
rp, ex, arch, ep, read, sha = old.rp, old.ex, old.arch, old.ep, old.read, old.sha
TRAIN = 88
BASE_ARMS = ('global_tree', 'weak_tree', 'global_ridge', 'weak_ridge')
RESIDUAL_ARMS = ('ridge_residual_global', 'ridge_residual_weak', 'ridge_cal_global', 'ridge_cal_state')
ARMS = BASE_ARMS + RESIDUAL_ARMS
PLACEMENTS = ('all', 'boundary')
BUDGETS = (.025, .05, .10)
MODEL_FILES = ('model.py', 'residual_calibration.py')


def dump(name, value):
    (P/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else x)+'\n')


def configs():
    rows = [dict(family='control', key='original65'), dict(family='control', key='incumbent66')]
    rows += [dict(family='conditional_model', arm=arm, placement=placement, budget=budget)
             for arm, placement, budget in itertools.product(ARMS, PLACEMENTS, BUDGETS)]
    return [dict(row, id=i) for i, row in enumerate(rows)]


def initialize():
    assert not (P/'PLAN.json').exists(), 'Do not overwrite preregistration'
    assert not (P/'FROZEN.json').exists(), 'Do not overwrite a frozen experiment'
    rows = configs()
    assert len(rows) == 50
    dump('PLAN.json', dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        objective='Research conditional monthly models toward70% annualized2016-08-17..2026-09-24; no guarantee or exactauthor claim.',
        grid='50fixed configs=original65+incumbent66+8arms*(all,boundary)*(.025,.05,.10). Expanded from26 to50 before initialization, fitting or training performance inspection. No further search or cross-family combinations.',
        arms=list(ARMS), placements=list(PLACEMENTS), budgets=list(BUDGETS),
        fixed='Same122 archived estimated historicalTop32+assumedETF17 pools, original295price/positive skipped longtrend eligibility, skip42 original50/52percent core. Monthly3equal/2half/<2cash, fixed adjusted units, no leverage, no intramonth changes.',
        inputs='Unchanged cached train_plain/event_plain10features and target_return from prior earnings/core research; no new earnings features, identity/date features, eligibility, target or label edits. Future-path validity is a label-only condition.',
        weak='At each signal, SPY adjustedclose<its trailing200-session simple mean OR fraction of that same historicalpool nonETFstocks strictly above their own trailing200-session means<.40. Full200 positivefinite prices required; breadth requires every supplied stock to be valid, no dropping unavailable names. An observed true clause establishesweak; otherwise any unavailable clause isunknown and correctionneutral. Both knownfalse meansnonweak. All8arms apply corrections only to eventweak; trainingweak repeats that historicalevent state across its49rows.',
        annual='PriorDec31 freeze, signal dates strictly inside preceding5years, only matured next-month labels, equal aggregate month weight before24month half-life decay, model-specific training-only standardization. Global models minimum18distinctmonths; specialized weak models independently require minimum6weakmonths and200weakrows. Insufficient forecasts remainunknown, no eligibilityfilter.',
        base_models='global_tree/weak_tree HistGradientBoosting loss=squared_error/max_iter100/learning_rate.05/max_depth2/min_samples_leaf50/l2_regularization1/random_state1729/early_stoppingFalse; matched global_ridge/weak_ridge alpha10 with training-only weighted standardization. Within each scope both models share rows and independently recomputed monthlybalanced24monthdecay weights normalized to admittedrowcount. Fixed parameters, no hyperparameter selection.',
        residual_models='RawRidge alpha10 with sameannual5year/priorDec31/maturedlabel/monthbalanced24month/min18months procedure. HistoricalOOF predictions use each historical signalyear ownpriorDec31 model; never in-sample residuals. Residual e=y-OOF only if OOFavailable and label matured. Global andweak residual trees require at least12weakmonths,2signalyears,240weakrows,Kish monthly effectiveN>=8; inadequate readiness contributes0. HistGradientBoosting depth2/maxleaf4/100iter/lr.05/minleaf40/l2=1/seed1729/noearly. Monthlybalanced24monthdecay. Addresidual onlyeventweak; unknownrawRidge remainsunknown.',
        calibration='HistoricalOOF pairs(p,y), weight eachmonth total=24monthdecay without normalizing to rowcount. b=clip(sum(w*p*y)/(sum(w*p*p)+6*weightedmean(p*p)),0,1). global andstate groups require12months,2years,240rows,KishmonthlyN>=8 and finite nonzero predictionsecondmoment. Coldglobal b=1; coldstate fallsbackreadyglobal then1. State is historicalweak boolean. MissingOOF never becomeszeroerror. Allcalibration corrections still eventweakonly.',
        correction='effect=.5*tanh(pred/.05). Original support=(base-core)/(1-coreweight); score=base+budget*(clip(support+effect,-.5,.5)-support). Missing/invalid forecast or nonweak event has exactlyzero effect. Original weighted core>=.5 unchanged. all reranks eligiblepool; boundary retains originaltop2 and contests third onlyamong originalranks3..5 within.02 oforiginalthird; explicit chosenmembership survives make_orders.',
        selection='First88 ending2023-11-30 only. Admit trainingCAGR>=incumbent66, Sharpe>=incumbent-.05, maxDD>=incumbent-.03, early53/recent35 CAGR each>=incumbent-.025. Highest admitted trainingCAGR primary; fallback66. Robust highest min(early,recent)Sharpe, thenCAGR,thenlowerID. Freeze besttraining diagnostic for each8arms regardlessadmission. No later34 promotion.',
        validation='Initialize registration before prepare/forecasts/train. Freeze selection/code/data beforeevaluate later34. Original controls parity, causalweakstate prefix tests, correction/boundary tests, independentfixed-unit dailyengine vs vectorpaths, trainingfreezeparity, nextopenzero and5bp/side unchangedorders, changedmonths andundo best1/3 incremental gains, chronologicalprefix36/60/76 diagnostics.',
        limitations='Previously exposed developmenthistory, not pristineOOS. Sameclose signal/fill is idealized; historicalpool/cap/sector proxies remain. Repeated target-seeking may overfit. No live observers/rules/accounts/ledgers/automations modified.',
        implementation_hashes={file: sha(P/file) for file in ('study.py',)+MODEL_FILES}))
    dump('GRID.json', rows)
    # Inherit the last verified protected snapshot; do not rescan or bless changes.
    dump('PROTECTED.json', read(previous.P/'PROTECTED.json'))


def assert_registered():
    plan = read(P/'PLAN.json')
    assert read(P/'GRID.json') == configs(), 'Registered grid differs from fixed implementation'
    old.assert_hashes(plan['implementation_hashes'], P)


def weak_states(panel, events, etfs):
    """Causal event states using exactly the supplied historical stock pool.

    Missing clause values never become a weak observation. A known true clause
    can establish the OR even when the other clause is missing. No stocks are
    dropped from the breadth denominator and ETFs never enter that denominator.
    """
    px = panel['adj']
    if (not px.index.is_unique or not px.index.is_monotonic_increasing
            or not px.columns.is_unique or 'SPY' not in px):
        raise ValueError('Need unique ordered price sessions/columns includingSPY')
    clean = px.where(np.isfinite(px) & px.gt(0))
    ma = clean.rolling(200, min_periods=200).mean()
    etfs = set(etfs)
    weak, records = [], []
    for day, supplied in events:
        day, symbols = pd.Timestamp(day), list(supplied)
        if day not in px.index or len(set(symbols)) != len(symbols) or not set(symbols).issubset(px.columns):
            raise ValueError('Missing event session/candidate or duplicatecandidate')
        stocks = [s for s in symbols if s not in etfs]
        spy, spy_ma = clean.at[day, 'SPY'], ma.at[day, 'SPY']
        spy_known = bool(np.isfinite(spy) and np.isfinite(spy_ma))
        spy_weak = bool(spy_known and spy < spy_ma)
        stock_prices, stock_ma = clean.loc[day, stocks], ma.loc[day, stocks]
        breadth_known = bool(stocks and np.isfinite(stock_prices).all() and np.isfinite(stock_ma).all())
        breadth = float((stock_prices > stock_ma).mean()) if breadth_known else None
        breadth_weak = bool(breadth_known and breadth < .40)
        active = spy_weak or breadth_weak
        known = active or (spy_known and breadth_known)
        weak.append(active)
        records.append(dict(date=str(day.date()), weak=active, known=bool(known),
            spy_clause_available=spy_known, spy_below_ma200=spy_weak if spy_known else None,
            breadth_available=breadth_known, stock_count=len(stocks),
            stock_above_ma200_fraction=breadth, breadth_below40=breadth_weak if breadth_known else None))
    return np.asarray(weak, dtype=bool), dict(events=len(records), weak_events=int(sum(weak)),
        unknown_events=sum(not x['known'] for x in records), etfs_excluded=True,
        full_stock_breadth_required=True, records=records)


def prepare():
    assert_registered()
    assert not (P/'FROZEN.json').exists()
    m, inherited, _, _, px, op = previous.load()
    keys = ('eligible', 'base_original65', 'core_original65', 'cw_original65', 'dates', 'ends',
            'event_dates', 'train_plain', 'event_plain', 'train_good', 'train_eligible', 'target_return')
    a = {key: inherited[key].copy() for key in keys}
    am = read(arch.P/'META.json')
    tm = am['training']
    assert m['dates'] == am['monthly']['dates'] and m['symbols'] == am['monthly']['symbols']
    for meta in (m, tm):
        assert all(len(ss) == 49 and len(set(ss)) == 49 and set(m['etfs']).issubset(ss) for ss in meta['symbols'])
    np.testing.assert_array_equal(np.asarray(a['dates'], dtype='datetime64[D]'),
        np.repeat(np.asarray(tm['dates'], dtype='datetime64[D]'), 49))
    np.testing.assert_array_equal(np.asarray(a['event_dates'], dtype='datetime64[D]'),
        np.repeat(np.asarray(m['dates'], dtype='datetime64[D]'), 49))
    assert a['train_plain'].shape == (len(a['dates']), 10)
    assert a['event_plain'].shape == (len(a['event_dates']), 10)
    panel, info = arch.old.panel_load(m['etfs'])
    train_weak, train_audit = weak_states(panel, zip(tm['dates'], tm['symbols']), m['etfs'])
    event_weak, event_audit = weak_states(panel, zip(m['dates'], m['symbols']), m['etfs'])
    a.update(train_weak=np.repeat(train_weak, 49), event_weak=np.repeat(event_weak, 49),
             monthly_weak=event_weak)
    actual = rp.make_orders(rp.choose(a['base_original65'], a['eligible'], m['symbols'], 3), a['base_original65'], m)
    assert actual == read(ep.P/'original65_ORDERS.json'), 'Original65 cachedorders changed'
    np.savez_compressed(P/'DATA.npz', **a)
    dump('META.json', m)
    dump('FEATURE_AUDIT.json', dict(training=train_audit, monthly=event_audit,
        original_eligibility_unchanged=True, feature_count=10, targets_unchanged=True))
    hashes = dict(info['source_hashes'])
    hashes.update(read(previous.P/'INPUT_HASHES.json'))
    hashes.update(read(old.P/'INPUT_HASHES.json'))
    for file in [previous.P/'DATA.npz', previous.P/'META.json', previous.P/'FROZEN.json',
        previous.P/'RESULTS.json', previous.P/'PROTECTED.json', Path(previous.__file__),
        old.P/'DATA.npz', old.P/'FORECASTS.npz', Path(old.__file__),
        arch.P/'META.json', ep.P/'primary_ORDERS.json', ep.P/'original65_ORDERS.json',
        Path(rp.__file__), Path(ex.e.engine.__file__), Path(old.session.__file__),
        O/'monthly_architecture_restart/residual_model.py'] + [P/x for x in ('study.py',)+MODEL_FILES]:
        hashes[str(file)] = sha(file)
    dump('INPUT_HASHES.json', hashes)
    dump('PREPARE_VALIDATION.json', dict(original65_orders_exact=True,
        original_eligibility_unchanged=True, original_features_and_targets_copied=True,
        monthly_periods=len(m['dates']), training_periods=len(tm['dates']), feature_count=10))
    print('Prepared fixed50-rule inputs and causal historicalweakstate', flush=True)


def forecasts():
    assert_registered()
    assert not (P/'FROZEN.json').exists()
    old.assert_hashes(read(P/'INPUT_HASHES.json'))
    a = dict(np.load(P/'DATA.npz'))
    shape = a['eligible'].shape
    args = (a['dates'], a['ends'], a['train_plain'], a['target_return'], a['train_good'],
            a['event_dates'], a['event_plain'], a['eligible'].reshape(-1), a['train_weak'], a['event_weak'])
    predictions, audits = {}, {}
    for filename, arms in zip(MODEL_FILES, (BASE_ARMS, RESIDUAL_ARMS)):
        model = module('conditional70_'+Path(filename).stem, P/filename)
        result = model.annual_forecasts(*args, min_global_months=18, min_weak_months=6, min_weak_rows=200)
        if isinstance(result, tuple):
            values, audit = result
        else:
            values, audit = result, result['audit']
        audits[Path(filename).stem] = audit
        for arm in arms:
            pred = np.asarray(values[arm], float)
            available = np.asarray(values['available_'+arm])
            assert pred.shape == available.shape == (shape[0]*shape[1],), arm
            assert available.dtype.kind == 'b', arm
            assert np.isfinite(pred[available]).all(), arm
            assert np.isnan(pred[~available]).all(), arm
            predictions['pred_'+arm] = pred.reshape(shape)
            predictions['available_'+arm] = available.reshape(shape)
    np.savez_compressed(P/'FORECASTS.npz', **predictions)
    dump('FORECAST_AUDIT.json', audits)
    dump('FORECAST_INPUT_HASHES.json', {file: sha(P/file) for file in
        ('PLAN.json', 'GRID.json', 'DATA.npz', 'study.py')+MODEL_FILES})
    print('Prepared8annual modelarms; missingforecasts remainunknown', flush=True)


def load():
    assert_registered()
    old.assert_hashes(read(P/'INPUT_HASHES.json'))
    old.assert_hashes(read(P/'FORECAST_INPUT_HASHES.json'), P)
    m, _, _, _, px, op = previous.load()
    assert read(P/'META.json') == m
    return m, {**dict(np.load(P/'DATA.npz')), **dict(np.load(P/'FORECASTS.npz'))}, px, op


def _core_weight(a, n, shape):
    cw = np.asarray(a['cw_original65'], float)
    if cw.ndim == 0:
        return np.broadcast_to(cw, shape)
    if cw.shape[0] < n:
        raise ValueError('Coreweight has insufficient eventrows')
    cw = cw[:n]
    if cw.ndim == 1:
        cw = cw[:, None]
    return np.broadcast_to(cw, shape)


def choices(row, a, m, n):
    """Pure prefix selection. Boundary membership is explicit, never reranked."""
    if row['family'] == 'control':
        if row['key'] == 'incumbent66':
            return old.incumbent_choices(m)[:n], None
        if row['key'] != 'original65':
            raise ValueError('Unknowncontrol')
        sc = a['base_original65'][:n].copy()
        return rp.choose(sc, a['eligible'][:n], m['symbols'][:n], 3), sc
    if (row['family'] != 'conditional_model' or row['arm'] not in ARMS
            or row['placement'] not in PLACEMENTS or row['budget'] not in (0.,)+BUDGETS):
        raise ValueError('Unregisteredmodelconfiguration')
    base = np.asarray(a['base_original65'], float)
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not 0 <= n <= len(base):
        raise ValueError('n must be an available nonnegativeeventprefix')
    base = base[:n]
    good = np.asarray(a['eligible'][:n])
    core = np.asarray(a['core_original65'][:n], float)
    cw = _core_weight(a, n, base.shape)
    pred = np.asarray(a['pred_'+row['arm']][:n], float)
    available = np.asarray(a['available_'+row['arm']][:n])
    weak = np.asarray(a['monthly_weak'][:n])
    symbols = m['symbols'][:n]
    if (good.dtype.kind != 'b' or available.dtype.kind != 'b' or weak.dtype.kind != 'b'
            or any(v.shape != base.shape for v in (good, core, pred, available)) or weak.shape != (n,)
            or len(symbols) != n or any(len(ss) != base.shape[1] or len(set(ss)) != len(ss) for ss in symbols)):
        raise ValueError('Misalignedmodelinputs')
    if (not np.isfinite(base[good]).all() or not np.isfinite(core[good]).all()
            or not np.isfinite(cw[good]).all() or (cw[good] < .5).any()
            or (cw[good] >= 1.).any() or (cw[good]+row['budget'] > 1.).any()):
        raise ValueError('Invalidscoresorcorebudget')
    ranks = [sorted(np.flatnonzero(g).tolist(), key=lambda j: (-b[j], ss[j]))
             for b, g, ss in zip(base, good, symbols)]
    usable = good & available & np.isfinite(pred) & weak[:, None]
    candidates = []
    if row['placement'] == 'boundary':
        boundary = np.zeros(base.shape, bool)
        for i, rank in enumerate(ranks):
            allowed = [j for j in rank[2:5] if base[i, rank[2]]-base[i, j] <= .02] if len(rank) >= 3 else []
            candidates.append(allowed)
            boundary[i, allowed] = True
        usable &= boundary
    effect = np.zeros(base.shape)
    effect[usable] = .5*np.tanh(pred[usable]/.05)
    changed = (effect != 0.) & (row['budget'] != 0.)
    sc = base.copy()
    support = (base[changed]-core[changed])/(1-cw[changed])
    sc[changed] += row['budget']*(np.clip(support+effect[changed], -.5, .5)-support)
    if row['placement'] == 'all':
        return rp.choose(sc, good, symbols, 3), sc
    selected = []
    for i, rank in enumerate(ranks):
        if len(rank) < 3:
            selected.append(rank if len(rank) == 2 else [])
        else:
            winner = min(candidates[i], key=lambda j: (-sc[i, j], symbols[i][j]))
            selected.append(sorted(rank[:2]+[winner], key=lambda j: (-sc[i, j], symbols[i][j])))
    return selected, sc


def admitted(z, incumbent):
    return previous.admitted(z, incumbent)


def select(rows):
    """Choose solely from provided training diagnostics; incumbent alwaysfallback."""
    incumbent = rows[1]
    for z in rows:
        z['admitted'] = admitted(z, incumbent)
        z['old_strict_admitted'] = old.session.admitted(z, incumbent)
    best = lambda xs: max(xs, key=lambda z: (z['training']['cagr'], -z['spec']['id']))
    qualified = [z for z in rows if z['admitted']]
    if not qualified:
        qualified = [incumbent]
    selected = dict(primary=best(qualified), robust=max(qualified, key=lambda z: (
        min(z['early']['sharpe'], z['recent']['sharpe']), z['training']['cagr'], -z['spec']['id'])),
        original65=rows[0], incumbent66=incumbent)
    for arm in ARMS:
        selected[arm+'_diagnostic'] = best([z for z in rows if z['spec'].get('arm') == arm])
    return selected


def train():
    assert_registered()
    assert not (P/'FROZEN.json').exists()
    m, a, px, _ = load()
    blocks, _, bounds, valid = rp.paths(px, m, TRAIN)
    rows, returns = [], []
    for spec in read(P/'GRID.json'):
        ch, _ = choices(spec, a, m, TRAIN)
        rr = rp.fast_returns(ch, blocks, valid)
        rows.append(dict(spec=spec, **old.session.measure(rr, m, bounds)))
        returns.append(rr)
    selected = select(rows)
    folds, stitched, control = [], [], []
    for n, end in ((36, 60), (60, 76), (76, 88)):
        prefix = [dict(spec=z['spec'], **old.session.measure(rr, m, bounds, n)) for z, rr in zip(rows, returns)]
        winner = select(prefix)['primary']
        rr = returns[winner['spec']['id']]
        stitched.extend(rr[bounds[n]:bounds[end]])
        control.extend(returns[1][bounds[n]:bounds[end]])
        folds.append(dict(train_end=m['ends'][n-1], test_end=m['ends'][end-1], spec=winner['spec']))
    dump('WALKFORWARD.json', dict(folds=folds,
        metrics=rp.stats_rr(np.array(stitched), m['dates'][36], m['ends'][87]),
        incumbent=rp.stats_rr(np.array(control), m['dates'][36], m['ends'][87]), pristine_oos=False))
    dump('TRAIN_TRIALS.json', rows)
    np.savez_compressed(P/'TRAIN_RETURNS.npz', returns=np.stack(returns), bounds=np.asarray(bounds))
    files = ('PLAN.json', 'GRID.json', 'DATA.npz', 'META.json', 'FEATURE_AUDIT.json', 'INPUT_HASHES.json',
             'FORECASTS.npz', 'FORECAST_AUDIT.json', 'FORECAST_INPUT_HASHES.json', 'PROTECTED.json', 'study.py')+MODEL_FILES
    dump('FROZEN.json', dict(created_at=datetime.now(timezone.utc).isoformat(), selected=selected,
        new_admitted=sum(z['admitted'] and z['spec']['family'] != 'control' for z in rows),
        trials=len(rows), hashes={file: sha(P/file) for file in files}, pristine_oos=False,
        diagnostics_may_be_unqualified=True, author_labels_used=False))
    (P/'FROZEN.sha256').write_text(sha(P/'FROZEN.json')+'\n')
    print('Frozen primary:', selected['primary']['spec'], 'training', selected['primary']['training'], flush=True)


def evaluate():
    f = read(P/'FROZEN.json')
    assert sha(P/'FROZEN.json') == (P/'FROZEN.sha256').read_text().strip()
    old.assert_hashes(f['hashes'], P)
    m, a, px, op = load()
    blocks, _, bounds, valid = rp.paths(px, m, len(m['dates']))
    orders = {}
    for name, z in f['selected'].items():
        ch, sc = choices(z['spec'], a, m, len(m['dates']))
        orders[name] = read(ep.P/'primary_ORDERS.json') if sc is None else rp.make_orders(ch, sc, m)
    orders['qqq'] = [dict(date=m['dates'][0], signal_date=m['dates'][0], end=m['ends'][-1], symbols=['QQQ'], scores=[])]
    results, memo = {}, {}
    for name, oo in orders.items():
        key = json.dumps(oo, sort_keys=True)
        if key not in memo:
            met, eq, tr, hold = ex.e.engine.replay(px, op, oo, cost=0., execution='close',
                start=pd.Timestamp(m['dates'][0]), end=pd.Timestamp(m['ends'][-1]))
            for row in met['annual_returns']:
                if row['year'] in (2016, 2026): row['partial'] = True
            common = dict(metrics=met, slices=arch.old.prior.old.previous.cuts(eq, m))
            for label, cost in (('next_open_zero', 0.), ('next_open_cost', .0005)):
                mm, ee, _, _ = ex.e.engine.replay(px, op, oo, cost=cost, execution='open',
                    start=pd.Timestamp(m['dates'][0]), end=pd.Timestamp(m['ends'][-1]))
                for row in mm['annual_returns']:
                    if row['year'] in (2016, 2026): row['partial'] = True
                common[label] = dict(metrics=mm, slices=arch.old.prior.old.previous.cuts(ee, m))
            memo[key] = common, eq, tr, hold
        common, eq, tr, hold = memo[key]
        if name != 'qqq':
            ch = [[m['symbols'][i].index(s) for s in row['symbols']] for i, row in enumerate(oo)]
            rr = rp.fast_returns(ch, blocks, valid)
            np.testing.assert_allclose(eq, np.r_[1., np.cumprod(1+rr)]*100000, rtol=1e-11, atol=1e-5)
            assert abs(common['slices']['training2016_2023']['cagr']-f['selected'][name]['training']['cagr']) < 1e-10
        results[name] = dict(common, selection=f['selected'].get(name))
        for suffix, value in (('ORDERS', oo), ('DAILY', [dict(date=str(t.date()), equity=float(v)) for t, v in eq.items()]),
                              ('TRADES', tr), ('HOLDINGS', hold)):
            dump(name+'_'+suffix+'.json', value)
        print(name, round(common['metrics']['cagr']*100, 3), round(common['metrics']['sharpe'], 3), flush=True)
    prior_results = read(previous.P/'RESULTS.json')
    for name in ('original65', 'incumbent66'):
        for key in ('cagr', 'sharpe', 'max_drawdown'):
            assert abs(results[name]['metrics'][key]-prior_results[name]['metrics'][key]) < 1e-10
    months = []
    for i, day in enumerate(m['dates']):
        row = dict(date=day, end=m['ends'][i], weak=bool(a['monthly_weak'][i]),
                   split='early53' if i < 53 else 'recent35' if i < 88 else 'later34')
        for name, oo in orders.items():
            if name == 'qqq': continue
            ss = oo[i]['symbols']
            r = px.loc[m['ends'][i], ss]/px.loc[day, ss]-1 if ss else []
            row[name] = dict(symbols=ss, monthly_return=float(np.mean(r)) if len(r) else 0.)
        months.append(row)
    dump('MONTH_COMPARISON.json', months)
    attribution = {}
    for name in f['selected']:
        if name in ('original65', 'incumbent66'): continue
        changes = []
        for row in months:
            x, b = row[name], row['incumbent66']
            if set(x['symbols']) != set(b['symbols']):
                changes.append(dict(date=row['date'], split=row['split'], weak=row['weak'],
                    old=b['symbols'], new=x['symbols'], difference=x['monthly_return']-b['monthly_return'],
                    log_contribution=float(np.log1p(x['monthly_return'])-np.log1p(b['monthly_return']))))
        positive = sorted([z['log_contribution'] for z in changes if z['log_contribution'] > 0], reverse=True)
        met = results[name]['metrics']
        undo = lambda n: float(((1+met['total_return'])*np.exp(-sum(positive[:n])))**(1/met['years'])-1)
        attribution[name] = dict(changed_months=len(changes), changes=changes,
            undo_best1_diagnostic_cagr=undo(1), undo_best3_diagnostic_cagr=undo(3))
    dump('ATTRIBUTION.json', attribution)
    protected = read(P/'PROTECTED.json')
    changed = [file for file, digest in protected.items() if sha(file) != digest]
    assert not changed, changed
    dump('RESULTS.json', results)
    dump('VALIDATION.json', dict(original_controls_exact=True, independent_fixed_unit_replay=True,
        training_freeze_parity=True, protected_files=len(protected), protected_changes=changed,
        observer_accounts_changed=False, pristine_oos=False, fixed_configs=50,
        primary_target70_reached=results['primary']['metrics']['cagr'] >= .70))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=('initialize', 'prepare', 'forecasts', 'train', 'evaluate'))
    globals()[ap.parse_args().phase]()
