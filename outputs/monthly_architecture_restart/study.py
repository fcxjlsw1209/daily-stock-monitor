"""Independent monthly score architectures with genuinely historical residual ML.

Research-only writes in this directory. No observer, pool or ledger changes.
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
sys.path.insert(0, str(O / 'monthly_state_rank_research'))
import state_rank as old
from residual_model import annual_forecasts

rp, ex, read, sha = old.rp, old.ex, old.read, old.sha
TRAIN = 88
FAMILIES = ['raw', 'aligned_sharpe', 'raw_vol', 'raw_sector', 'recent_risk']
NORMS = ['all', 'eligible']
AUX = ['cmf21', 'cmf63', 'signed_volume', 'volume_ratio', 'close_location21', 'compression']
ARMS = [('none', 0., 0.), ('hand', .05, 0.), ('direct', 0., .05),
        ('direct', 0., .10), ('residual', 0., .05), ('residual', 0., .10),
        ('direct', .05, .05), ('residual', .05, .05)]


def dump(name, value):
    (P / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False, default=lambda v: v.item() if isinstance(v, np.generic) else v) + '\n')


def architectures():
    return [dict(id=i, family=f, normalization=n) for i, (f, n) in
            enumerate(itertools.product(FAMILIES, NORMS))]


def grid():
    rows = []
    for base, (arm, hand, budget) in itertools.product(architectures(), ARMS):
        rows.append(dict(id=len(rows), architecture=base, arm=arm,
                         hand_budget=hand, ml_budget=budget))
    return rows


def initialize():
    assert not (P / 'PLAN.json').exists()
    dump('PLAN.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Independent simple monthly architectures; no top-two lock; compare raw-target versus causally cross-fitted trend-residual ML.',
        universe='Unchanged122 cached Top32-estimate+assumed theme17 pools. Positive P[t-42]/P[t-252]-1 and295completeprices; no current pullback requirement. Signals2016-08-17 initialization then monthends through2026-08-31; last valuation2026-09-24. Initial partial-month convention is not recovered author trading.',
        baseline='Five definitions: rank210sessionreturn ending42sessionsago; mean/std of same210returns; .75raw+.25vol126; .75raw+.25mean own-sector126/252 relative; rawreturn/recent63vol comparator. Two cross-sectional rank sets:all observed members or past-eligible only. Missing auxiliary features and ETF-sector relative are neutral0. Eligibility unchanged across10bases.',
        correction='Eight arms per base: none;5%fixed hand basket; direct/residual model at5/10%; hand5%plusdirect/residual5%. Hand=.4ownsector+.2CMF21+.2signedvol63+.2mean21daycloselocation. Score=(1-hand)*base+hand*handbasket+mlbudget*.5*tanh(predictedreturn/.05). No rank expansion of tiny ML predictions. Combined nominal addonbudget<=10%; all eligible names compete for all3slots.',
        ml='Native next-calendar-month clipped excessSPY labels, centered among valid eligible training names in that month. No author holdings, date or ticker features. Base mapping uses this architecture score plus rawmomentum/volatility/sector ranks and score squared; auxiliary six volume/price ranks plus breadth/trend/highvol interactions. Fixed alpha10,24monthhalf-life, equal total month weight before age decay, past-only standardization.',
        residual='For each historical month base prediction fitted only on earlier signals with label_end<=that signaldate, prior5years, at least18months. Residual=y-baseOOF. Annual aux Ridge fits preceding5years completed OOF residuals; direct comparator uses exact same rows and features but y. Calendar signal-year annual cutoff previousDec31. Earliest sparse months defaultzero. Trading eligibility never uses future-label validity.',
        training_data='Reuse monthend identities/pool origins from monthly_blend_optimization/MONTHLY_TRAIN.json,2011..2025. BeforeAug2016 membership is inherited historical proxy; later latest known monthly archive. Recompute past-only eligibility/features. Cached good is used ONLY to decide matured training labels.',
        selection='80fixedconfigs; train first88 through2023-11-30. New-primary highest trainingCAGR with Sharpe>=1,DD>=-.45,each early53/recent35 Sharpe>=.5. Robust highest min(early,recent)Sharpe then trainingCAGR. No requirement to beat previously optimized66.36 in every segment. If no config passes, mark no_admissible and retain descriptive robust only. Freeze before last34 replay.',
        controls='Freeze best noML/directed/residual families and unmodified per-family all-normalization bases as diagnostics. Also original65.34 and previous66.36 exact122 orders and QQQ. Prefix36/60/76 selection inside training diagnostics. No further grid after validation.',
        execution='Long only, unlevered3equal/2half/<2cash, monthly fixed adjusted-return units. Same-final-close0cost idealized primary, next-open0cost and5bps/side on actual turnover sensitivity. Models remain research, never ledger trades.',
        validation='Feature prefix tests; skip42alignment; same eligibility; OOF and annual labels mature; zero forecast exactnochange; all slots free; bounded corrections; fast daily returns match fixed-units engine; protected file hashes.',
        limits='Previously repeatedly explored history, not pristine OOS. Historical cap/sector proxies and unknown author ETF17 remain. No guaranteed80% or exact-author reproduction. No changes to existing strategies or monitoring.'))
    dump('GRID.json', grid())
    protected = read(O / 'monthly_daily_episodes/PROTECTED.json')
    for folder in ['monthly_daily_episodes', 'monthly_stability_monitor', 'paired_stock_monitor']:
        protected.update({str(f): sha(f) for f in (O / folder).rglob('*')
                          if f.is_file() and f.suffix != '.pyc' and f.name != 'run.lock'})
    dump('PROTECTED.json', protected)


def rank_values(values, eligible=None):
    s = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan)
    if eligible is not None:
        s = s.where(eligible)
    return s.rank(pct=True).fillna(.5).to_numpy() - .5


def raw_frames(panel):
    px = panel['adj']
    r = px.pct_change(fill_method=None)
    mom = {n: px.shift(42) / px.shift(n) - 1 for n in [126, 252]}
    v63 = r.rolling(63).std()
    span = (panel['high'] - panel['low']).replace(0, np.nan)
    loc = (2 * panel['close'] - panel['high'] - panel['low']) / span
    volume = panel['volume']
    f = dict(raw=mom[252], aligned_sharpe=r.rolling(210).mean().shift(42) /
             r.rolling(210).std().shift(42).replace(0, np.nan),
             recent_risk=mom[252] / v63.replace(0, np.nan), vol=r.rolling(126).std(),
             signed_volume=(np.sign(r) * volume).rolling(63).sum() / volume.rolling(63).sum().replace(0, np.nan),
             volume_ratio=volume.rolling(21).mean() / volume.rolling(126).mean().replace(0, np.nan),
             close_location21=loc.rolling(21).mean(),
             compression=-r.rolling(21).std() / r.rolling(126).std().replace(0, np.nan))
    for n in [21, 63]:
        f['cmf' + str(n)] = (loc * volume).rolling(n).sum() / volume.rolling(n).sum().replace(0, np.nan)
    good = px.notna().rolling(295).sum().eq(295) & px.gt(0) & mom[252].gt(0)
    return f, mom, good


def snapshot(day, ss, f, mom, good, panel, etfs, norm):
    eligible = good.loc[day, ss].to_numpy()
    use = eligible if norm == 'eligible' else None
    rank = lambda z: rank_values(z, use)
    row = {k: rank(v.loc[day, ss]) for k, v in f.items()}
    bm = ex.t.strict.benchmark_map(day, ss, panel['sectors'], etfs)
    row['sector'] = sum(rank(ex.t.strict.relative_row(mom[n], day, ss, bm)) for n in [126, 252]) / 2
    row['raw_vol'] = .75 * row['raw'] + .25 * row['vol']
    row['raw_sector'] = .75 * row['raw'] + .25 * row['sector']
    row['hand'] = .4 * row['sector'] + .2 * row['cmf21'] + .2 * row['signed_volume'] + .2 * row['close_location21']
    state = old.state_row(panel['adj'], day, ss, etfs)
    z = np.column_stack([row[k] for k in AUX])
    states = [2 * state['breadth'] - 1, 2 * float(state['trend']) - 1, 2 * float(state['highvol']) - 1]
    aux = np.column_stack([z] + [z * s for s in states])
    return row, eligible, aux


def prepare():
    m, d, *_ = old.prior.load()
    panel, info = old.panel_load(m['etfs'])
    f, mom, good = raw_frames(panel)
    tm = read(O / 'monthly_blend_optimization/MONTHLY_TRAIN.json')
    cached = dict(np.load(O / 'monthly_blend_optimization/MONTHLY_TRAIN.npz'))
    dates = np.repeat(np.array(tm['dates'], dtype='datetime64[D]'), np.diff(cached['offsets']))
    ends = np.repeat(np.array(tm['ends'], dtype='datetime64[D]'), np.diff(cached['offsets']))
    event_dates = np.repeat(np.array(m['dates'], dtype='datetime64[D]'), 49)
    arrays = dict(dates=dates, ends=ends, event_dates=event_dates, offsets=cached['offsets'])
    for norm in NORMS:
        train_rows, event_rows, train_aux, event_aux, gs, es = [], [], [], [], [], []
        for day, ss in zip(tm['dates'], tm['symbols']):
            row, g, a = snapshot(day, ss, f, mom, good, panel, set(m['etfs']), norm)
            train_rows.append(row); train_aux.append(a); gs.append(g)
        for i, (day, ss) in enumerate(zip(m['dates'], m['symbols'])):
            row, g, a = snapshot(day, ss, f, mom, good, panel, set(m['etfs']), norm)
            np.testing.assert_array_equal(g, d['eligible'][i])
            event_rows.append(row); event_aux.append(a); es.append(g)
        past_good = np.concatenate(gs)
        label_good = past_good & cached['good'] & np.isfinite(cached['Y'])
        y = cached['Y'].copy()
        for lo, hi in zip(cached['offsets'][:-1], cached['offsets'][1:]):
            g = label_good[lo:hi]
            if g.any():
                y[lo:hi] -= np.mean(y[lo:hi][g])
        arrays['y'] = y
        arrays['train_good'] = label_good
        arrays['eligible'] = np.stack(es)
        arrays['aux_' + norm] = np.vstack(train_aux)
        arrays['event_aux_' + norm] = np.vstack(event_aux)
        for key in FAMILIES + ['vol', 'sector', 'hand']:
            arrays['train_' + norm + '_' + key] = np.concatenate([r[key] for r in train_rows])
            arrays['event_' + norm + '_' + key] = np.stack([r[key] for r in event_rows])
        print('Prepared normalization', norm, flush=True)
    np.savez_compressed(P / 'DATA.npz', **arrays)
    dump('META.json', dict(monthly=m, training=tm, auxiliary_features=AUX,
                          auxiliary_interactions=['raw', 'breadth', 'SPY200trend', 'highvol']))
    hashes = info['source_hashes']
    for file in [O / 'monthly_blend_optimization/MONTHLY_TRAIN.npz',
                 O / 'monthly_blend_optimization/MONTHLY_TRAIN.json',
                 ex.P / 'INPUTS.json', ex.P / 'DATA.npz', ex.P / 'PRICES.npz',
                 Path(old.__file__), Path(rp.__file__), Path(ex.e.engine.__file__)]:
        hashes[str(file)] = sha(file)
    dump('INPUT_HASHES.json', hashes)
    dump('PREPARE_VALIDATION.json', dict(eligibility_parity=True, monthly_periods=122,
        training_months=len(tm['dates']), future_label_validity_never_filters_trading=True,
        candidate_counts=arrays['eligible'].sum(axis=1).tolist(), label_examples=int(arrays['train_good'].sum())))


def features_for_base(a, base, event=False):
    p = ('event_' if event else 'train_') + base['normalization'] + '_'
    s = a[p + base['family']].reshape(-1)
    return np.column_stack([s, s ** 2, a[p + 'raw'].reshape(-1),
                            a[p + 'vol'].reshape(-1), a[p + 'sector'].reshape(-1)])


def forecasts():
    a = dict(np.load(P / 'DATA.npz'))
    forecasts = {}
    logs = {}
    for base in architectures():
        norm = base['normalization']
        pred = annual_forecasts(a['dates'], a['ends'], features_for_base(a, base),
            a['aux_' + norm], a['y'], a['train_good'], a['event_dates'],
            features_for_base(a, base, True), a['event_aux_' + norm], a['eligible'].reshape(-1),
            alpha=10., half_life_months=24., min_months=18)
        for key in ['direct', 'residual']:
            forecasts[str(base['id']) + '_' + key] = pred[key].reshape(122, 49)
        logs[str(base['id'])] = pred['audit']
        print('Forecasts architecture', base, flush=True)
    np.savez_compressed(P / 'FORECASTS.npz', **forecasts)
    dump('FORECAST_AUDIT.json', logs)


def bounded_prediction(pred):
    return .5 * np.tanh(pred / .05)


def score(row, a, predictions, n):
    base = row['architecture']
    prefix = 'event_' + base['normalization'] + '_'
    result = ((1 - row['hand_budget']) * a[prefix + base['family']][:n]
              + row['hand_budget'] * a[prefix + 'hand'][:n])
    if row['ml_budget']:
        result = result + row['ml_budget'] * bounded_prediction(predictions[str(base['id']) + '_' + row['arm']][:n])
    return result


def choices(row, a, pred, m, n):
    sc = score(row, a, pred, n)
    return rp.choose(sc, a['eligible'][:n], m['symbols'][:n], 3), sc


def load():
    for p, h in read(P / 'INPUT_HASHES.json').items():
        assert sha(p) == h, p
    m, d, L, x, px, op = ex.load()
    return m, dict(np.load(P / 'DATA.npz')), dict(np.load(P / 'FORECASTS.npz')), px, op


def metrics(rr, m, bounds, n=TRAIN):
    whole = rp.stats_rr(rr[:bounds[n]], m['dates'][0], m['ends'][n - 1])
    cut = min(n // 2, 53) if n < TRAIN else 53
    early = rp.stats_rr(rr[:bounds[cut]], m['dates'][0], m['ends'][cut - 1])
    late = rp.stats_rr(rr[bounds[cut]:bounds[n]], m['dates'][cut], m['ends'][n - 1])
    return dict(training=whole, early=early, recent=late,
                robust=min(early['sharpe'], late['sharpe']))


def admissible(z):
    return (z['training']['sharpe'] >= 1. and z['training']['max_drawdown'] >= -.45
            and z['early']['sharpe'] >= .5 and z['recent']['sharpe'] >= .5)


def pick(rows, robust=False):
    admitted = [z for z in rows if admissible(z)]
    pool = admitted or rows
    key = ((lambda z: (z['robust'], z['training']['cagr'], -z['spec']['id'])) if robust or not admitted
           else (lambda z: (z['training']['cagr'], z['robust'], -z['spec']['id'])))
    return max(pool, key=key), bool(admitted)


def train():
    assert not (P / 'FROZEN.json').exists()
    m, a, pred, px, op = load()
    blocks, _, bounds, valid = rp.paths(px, m, TRAIN)
    rows, returns = [], []
    for spec in grid():
        ch, _ = choices(spec, a, pred, m, TRAIN)
        rr = rp.fast_returns(ch, blocks, valid)
        rows.append(dict(spec=spec, **metrics(rr, m, bounds)))
        returns.append(rr)
    primary, ok = pick(rows)
    robust, _ = pick(rows, True)
    selected = dict(primary=primary, robust=robust)
    for arm in ['none', 'hand', 'direct', 'residual']:
        subset = [z for z in rows if z['spec']['arm'] == arm]
        selected['best_' + arm] = pick(subset)[0]
    for family in FAMILIES:
        selected['simple_' + family] = next(z for z in rows if z['spec']['arm'] == 'none'
            and z['spec']['architecture'] == dict(id=FAMILIES.index(family) * 2, family=family, normalization='all'))
    # Exact paired ablations of the primary, determined BEFORE later34 replay.
    spec = primary['spec']
    for arm in ['none', 'hand', 'direct', 'residual']:
        hand = spec['hand_budget'] if arm in ['direct', 'residual'] else (.05 if arm == 'hand' else 0.)
        match = [z for z in rows if z['spec']['architecture'] == spec['architecture']
                 and z['spec']['arm'] == arm and z['spec']['hand_budget'] == hand
                 and z['spec']['ml_budget'] == (max(.05, spec['ml_budget']) if arm in ['direct', 'residual'] else 0.)]
        selected['paired_' + arm] = match[0]
    wf, pieces = [], []
    for n, end in [(36, 60), (60, 76), (76, 88)]:
        prefix = [dict(spec=z['spec'], **metrics(rr, m, bounds, n)) for z, rr in zip(rows, returns)]
        winner, guard = pick(prefix)
        i = winner['spec']['id']
        piece = returns[i][bounds[n]:bounds[end]]
        pieces.extend(piece)
        wf.append(dict(train_end=m['ends'][n-1],test_end=m['ends'][end-1],spec=winner['spec'],guard_pass=guard,
                       metrics=rp.stats_rr(piece,m['dates'][n],m['ends'][end-1])))
    dump('WALKFORWARD.json', dict(folds=wf, metrics=rp.stats_rr(np.array(pieces),m['dates'][36],m['ends'][87]),pristine_oos=False))
    dump('TRAIN_TRIALS.json', rows)
    files = ['PLAN.json', 'GRID.json', 'DATA.npz', 'META.json', 'INPUT_HASHES.json',
             'FORECASTS.npz', 'FORECAST_AUDIT.json', 'study.py', 'residual_model.py']
    dump('FROZEN.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),selected=selected,
         primary_admitted=ok, status='admitted' if ok else 'no_admissible_descriptive_robust_only',
         trials=len(rows), admissible=sum(admissible(z) for z in rows),
         hashes={f:sha(P/f) for f in files}, author_labels_used=False))
    (P / 'FROZEN.sha256').write_text(sha(P / 'FROZEN.json') + '\n')
    print('Frozen', {k:(v['spec']['id'],v['training']['cagr']) for k,v in selected.items()}, flush=True)


def evaluate():
    frozen = read(P / 'FROZEN.json')
    assert sha(P / 'FROZEN.json') == (P / 'FROZEN.sha256').read_text().strip()
    for f,h in frozen['hashes'].items():
        assert sha(P/f) == h, f
    m,a,pred,px,op = load()
    blocks,_,bounds,valid = rp.paths(px,m,122)
    all_orders,results = {},{}
    for name,z in frozen['selected'].items():
        ch,sc = choices(z['spec'],a,pred,m,122)
        all_orders[name] = rp.make_orders(ch,sc,m)
    for name,path in [('original65',O/'monthly_state_rank_research/baseline_ORDERS.json'),
                      ('previous66',O/'monthly_daily_episodes/primary_ORDERS.json')]:
        all_orders[name] = read(path)
    all_orders['qqq'] = [dict(date=m['dates'][0],signal_date=m['dates'][0],end=m['ends'][-1],symbols=['QQQ'],scores=[])]
    for name,orders in all_orders.items():
        met,eq,tr,hold = ex.e.engine.replay(px,op,orders,cost=0.,execution='close',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
        if name != 'qqq':
            ch = [[m['symbols'][i].index(s) for s in row['symbols']] for i,row in enumerate(orders)]
            rr = rp.fast_returns(ch,blocks,valid)
            np.testing.assert_allclose(eq,np.r_[1.,np.cumprod(1+rr)]*100000,rtol=1e-11,atol=1e-5)
        slices = old.prior.old.previous.cuts(eq,m)
        results[name] = dict(metrics=met,slices=slices,spec=frozen['selected'].get(name,{}).get('spec'))
        if name in frozen['selected']:
            assert abs(slices['training2016_2023']['cagr']-frozen['selected'][name]['training']['cagr']) < 1e-10
        for label,cost in [('next_open_zero',0.),('next_open_cost',.0005)]:
            mm,ee,_,_ = ex.e.engine.replay(px,op,orders,cost=cost,execution='open',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
            results[name][label] = dict(metrics=mm,slices=old.prior.old.previous.cuts(ee,m))
        for suffix,data in [('ORDERS',orders),('DAILY',[dict(date=str(day.date()),equity=float(v)) for day,v in eq.items()]),('TRADES',tr),('HOLDINGS',hold)]:
            dump(name+'_'+suffix+'.json',data)
        print(name, 'CAGR',round(met['cagr']*100,3),'Sharpe',round(met['sharpe'],4),flush=True)
    for name,key in [('original65','original65'),('previous66','primary')]:
        reference=read(O/'monthly_daily_episodes/RESULTS.json')[key]['metrics']
        for k in ['cagr','sharpe','max_drawdown']:
            assert abs(results[name]['metrics'][k]-reference[k]) < 1e-10
    comparison=[]
    for i,day in enumerate(m['dates']):
        row=dict(date=day,end=m['ends'][i],split='early53' if i<53 else 'recent35' if i<88 else 'later34')
        for name,orders in all_orders.items():
            if name=='qqq':continue
            ss=orders[i]['symbols']
            row[name]=dict(symbols=ss,monthly_return=float((px.loc[m['ends'][i],ss]/px.loc[day,ss]-1).mean()) if ss else 0.)
        comparison.append(row)
    dump('MONTH_COMPARISON.json',comparison)
    attribution={}
    for name in ['paired_none','paired_direct','previous66']:
        changes=[]
        for row in comparison:
            p,b=row['primary'],row[name]
            if set(p['symbols']) != set(b['symbols']):
                changes.append(dict(date=row['date'],end=row['end'],split=row['split'],old=b['symbols'],new=p['symbols'],
                    diff=p['monthly_return']-b['monthly_return'],
                    log_contribution=float(np.log1p(p['monthly_return'])-np.log1p(b['monthly_return'])),
                    original_top2_lost=len(set(b['symbols'][:2])-set(p['symbols']))))
        attribution[name]=dict(changed_months=len(changes),top2_changes=sum(z['original_top2_lost']>0 for z in changes),changes=changes)
    dump('ATTRIBUTION.json',attribution)
    changed=[p for p,h in read(P/'PROTECTED.json').items() if sha(p)!=h]
    assert not changed,changed
    dump('RESULTS.json',results)
    dump('VALIDATION.json',dict(independent_daily_replay=True,training_freeze_parity=True,old_controls_exact=True,
         protected_files=len(read(P/'PROTECTED.json')),protected_changes=changed,pristine_oos=False))


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=['initialize','prepare','forecasts','train','evaluate'])
    globals()[ap.parse_args().phase]()
