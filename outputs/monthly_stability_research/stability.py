"""Predeclared long-momentum architecture search on the extended monthly pool."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, itertools, json, sys
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
O = P.parent
sys.path.insert(0, str(O / 'monthly_2016_extension'))
import extend as ex
import optimize as previous
rp, read, sha = ex.rp, ex.read, ex.sha
TRAIN = 88


def dump(name, value):
    (P / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else x) + '\n')


def initialize():
    assert not (P / 'PLAN.json').exists()
    folders = ['monthly_2016_extension', 'monthly_industry_monitor',
        'monthly_regime_monitor', 'monthly_buffer5_monitor', 'daily_stock_monitor',
        'pressure_shadow_monitor', 'resilience_shadow_monitor', 'accel_shadow_monitor',
        'high52_shadow_monitor', 'accel40_shadow_monitor']
    dump('PROTECTED.json', {str(f): sha(f) for folder in folders for f in
        (O / folder).rglob('*') if f.is_file() and f.suffix != '.pyc' and f.name != 'run.lock'})
    dump('PLAN.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        hypothesis='Reconsider whole score architecture, not just tiny weight nudges. Long momentum skips42sessions; compare recent/aligned volatility risk adjustment, own-sector relative strength, path persistence and market-conditioned volatility preferences.',
        data='Unmodified122dates/Top32-proxy+assumed17ETF from2016 extension. No pool optimization; identical missing-data/positive252skip42/295warmup gate.',
        split='Train first88 periods Aug2016-Nov2023; freeze before last34. Separate early53 and recent35 training diagnostics. Historical periods already reused, not pristine OOS.',
        search='12 predefined long-core definitions x3coreweights x5supports x4riskmodes x2retentions; baseline controls. Then top2training configurations per core family, with <=10%price-volume/recentRidge correction globally or highvol-only. All searches training-only.',
        selection='Primary maxCAGR with trainSharpe>=previous-.05, trainDD>=previous-.03, earlyCAGR>=previous early, recent35CAGR>=previous recent-.05. Robust control prioritizes minimum early/recent Sharpe plus aggregate CAGR and drawdown. Baseline always included.',
        validation='Prefix-only36/60/76-month selection on predefined stageA, then12/16/12month forward blocks entirely before outer holdout. Freeze both final selectors, report both. No holdout-driven extra sweep.',
        execution='Long-only,unlevered,3equal,2half,<2cash; fixed monthly units. Complete-close/same-close0cost primary; next-open0cost and5bp/side controls. No intramonth exits.',
        limitations='Author full chart80.64%/Sharpe1.867 reference only; pre2021author holdings absent. Cap proxy/ETF list/sector history limitations inherited. Selection on reused history may overfit; no promotion to observers.',
        sources=['https://www.nber.org/papers/w20439','https://www.nber.org/papers/w22208'],
        source_scope='Motivate state/risk diagnostics only. These papers do not validate this long-only monthly strategy or predict its return.'))


def rank_rows(panel, m):
    return np.stack([ex.t.ranks(panel.loc[day, ss].replace([np.inf, -np.inf], np.nan))
        for day, ss in zip(m['dates'], m['symbols'])])


def features(px, m):
    r = px.pct_change(fill_method=None)
    vol = {n: r.rolling(n).std() for n in [63, 126, 252]}
    mom = {n: px.shift(42) / px.shift(n) - 1 for n in [252, 294]}
    f = {}
    for horizon in [252, 294]:
        prefix = str(horizon)
        f[prefix + '_raw'] = rank_rows(mom[horizon], m)
        for n, lag, power, key in [(63, 0, 1., 'v63'), (126, 0, 1., 'v126'),
                (252, 42, 1., 'aligned'), (63, 0, .5, 'half')]:
            denom = vol[n].shift(lag).pow(power).replace(0, np.nan)
            f[prefix + '_' + key] = rank_rows(mom[horizon] / denom, m)
    logret = np.log(px / px.shift(1))
    efficiency = (np.log(px.shift(42) / px.shift(294)) /
        logret.abs().rolling(252).sum().shift(42).replace(0, np.nan))
    f['path'] = rank_rows(efficiency, m)
    f['294_path'] = .8 * f['294_raw'] + .2 * f['path']
    f['blend_v63'] = .5 * f['252_v63'] + .5 * f['294_v63']
    f['accel'] = rank_rows(px.shift(42) / px.shift(126) - px.shift(126) / px.shift(210), m)
    f['vol'] = rank_rows(vol[126], m)
    f['downside'] = rank_rows(np.sqrt(r.clip(upper=0).pow(2).rolling(63).mean()), m)
    f['upside'] = rank_rows(np.sqrt(r.clip(lower=0).pow(2).rolling(126).mean()), m)
    f['upside_balance'] = rank_rows(np.sqrt(r.clip(lower=0).pow(2).rolling(126).mean()) -
        np.sqrt(r.clip(upper=0).pow(2).rolling(126).mean()), m)
    return f


def prepare():
    m, d, L, x, px, op = ex.load()
    f = features(px, m)
    np.testing.assert_allclose(f['294_v63'] * .52, d['core'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(f['vol'], x['vol126'], atol=1e-12, rtol=0)
    # Isolate the original state-CMF contribution without loading raw OHLCV anew.
    f['flow'] = (d['support'] - .219375 * f['vol'] - .185625 * f['accel']) / .075
    f['sector'] = .5 * x['sector126'] + .5 * x['sector252']
    f['stability'] = x['stability']
    f['tail'] = x['tail']
    np.savez_compressed(P / 'FEATURES.npz', **f)
    inputs = [ex.P / z for z in ['DATA.npz','INPUTS.json','PRICES.npz','FEATURES.npz',
        'LAYERS.npz','PREDICTIONS.npz','FROZEN.json','RESULTS.json']]
    inputs += [Path(ex.__file__), Path(previous.__file__), Path(rp.__file__), Path(ex.e.engine.__file__)]
    dump('INPUT_HASHES.json', {str(p): sha(p) for p in inputs})
    print('Prepared', len(m['dates']), 'events;', len(f), 'features; exact base parity', flush=True)


def load():
    for p, h in read(P / 'INPUT_HASHES.json').items():
        assert sha(p) == h, p
    return (*ex.load(), dict(np.load(P / 'FEATURES.npz')))


CORES = [h + '_' + k for h in ['252','294'] for k in ['raw','v63','v126','aligned','half']] + ['294_path','blend_v63']
SUPPORTS = {
    'legacy': [.45703125, .38671875, 0., 0., .15625],
    'relative': [.25, .30, .30, 0., .15],
    'balanced': [.30, .30, .20, .05, .15],
    'no_accel': [.40, 0., .35, .10, .15],
    'path': [.20, .20, .25, .20, .15],
}


def specs():
    rows = [dict(id=0, family='baseline', spec=dict(baseline=True))]
    for core, weight, support, risk, retain in itertools.product(CORES, [.50, .65, .80],
            SUPPORTS, ['static', 'highvol_lowrisk', 'weak_lowrisk', 'highvol_upside'], [3,5]):
        rows.append(dict(id=len(rows), family=core, spec=dict(core=core, core_weight=weight,
            support=support, risk=risk, retain=retain, addon='none', addon_budget=0., addon_mode='global')))
    return rows


def score(spec, m, d, L, x, f, n):
    if spec.get('baseline'):
        old = read(ex.P / 'FROZEN.json')['selected']['return_primary']['spec']
        return previous.score(old, d, L, x, n)
    core = f[spec['core']][:n]
    vol = f['vol'][:n].copy()
    if spec['risk'] in ['highvol_lowrisk','weak_lowrisk']:
        active = d['high'][:n] if spec['risk'] == 'highvol_lowrisk' else d['weak'][:n]
        vol = np.where(active[:,None], -f['downside'][:n], vol)
    elif spec['risk'] == 'highvol_upside':
        vol = np.where(d['high'][:n,None], f['upside_balance'][:n], vol)
    a,b,c,g,h = SUPPORTS[spec['support']]
    side = a*vol + b*f['accel'][:n] + c*f['sector'][:n] + g*f['path'][:n] + h*f['flow'][:n]
    w = spec['core_weight']
    budget = spec['addon_budget'] * (d['high'][:n].astype(float)
        if spec['addon_mode'] == 'highvol' else np.ones(n))
    ans = w*core + (1-w-budget[:,None])*side
    if spec['addon'] != 'none':
        addon = L[spec['addon']]['score'][:n] if spec['addon'] in L else f[spec['addon']][:n]
        ans += budget[:,None]*addon
    return ans


def decisions(spec, m, d, L, x, f, n):
    sc = score(spec,m,d,L,x,f,n)
    return rp.choose(sc,d['eligible'][:n],m['symbols'][:n],spec.get('retain',3)),sc


def measure(row, m, d, L, x, f, blocks, valid, bounds, n=TRAIN):
    ch, _ = decisions(row['spec'],m,d,L,x,f,n)
    rr = rp.fast_returns(ch,blocks[:n],valid[:n])
    met = rp.stats_rr(rr,m['dates'][0],m['ends'][n-1])
    cut = min(53,n)
    early = rp.stats_rr(rr[:bounds[cut]],m['dates'][0],m['ends'][cut-1])
    recent = rp.stats_rr(rr[bounds[53]:],m['dates'][53],m['ends'][n-1]) if n>53 else early
    robust = min(early['sharpe'],recent['sharpe']) + .3*met['sharpe'] + .4*np.log1p(met['cagr']) - .3*abs(met['max_drawdown'])
    return dict(row,training=met,early=early,recent=recent,robust=float(robust)),rr


def acceptable(row, base):
    return (row['training']['sharpe'] >= base['training']['sharpe']-.05 and
        row['training']['max_drawdown'] >= base['training']['max_drawdown']-.03 and
        row['early']['cagr'] >= base['early']['cagr']-1e-12 and
        row['recent']['cagr'] >= base['recent']['cagr']-.05)


def train():
    assert not (P/'FROZEN.json').exists()
    m,d,L,x,px,op,f = load()
    blocks, dates, bounds, valid = rp.paths(px,m,TRAIN)
    rows, returns = [], []
    for row in specs():
        z,rr = measure(row,m,d,L,x,f,blocks,valid,bounds)
        rows.append(z);returns.append(rr)
        if len(rows)%400==0: print('Training architectures',len(rows),flush=True)
    base = rows[0]; fixed_count=len(rows)
    seeds=[]
    for family in CORES:
        good=[r for r in rows if r['family']==family and acceptable(r,base)]
        seeds += sorted(good,key=lambda z:z['training']['cagr'],reverse=True)[:2]
    for seed,addon,budget,mode in itertools.product(seeds,
            ['anti_chase','volume_recovery','pressure_easing','ridge_10_12','tail','stability'],
            [.025,.05,.10],['global','highvol']):
        spec=dict(seed['spec'],addon=addon,addon_budget=budget,addon_mode=mode)
        z,rr=measure(dict(id=len(rows),family='small_adjustment',parent=seed['id'],spec=spec),
            m,d,L,x,f,blocks,valid,bounds)
        rows.append(z);returns.append(rr)
    good=[r for r in rows if acceptable(r,base)]
    selected={'return_primary':max(good,key=lambda z:(z['training']['cagr'],-z['id'])),
        'robust_control':max(good,key=lambda z:(z['robust'],-z['id']))}
    # Causal prefix selection on only predeclared stageA, not full-training-selected layer seeds.
    wf=[]; stitched=[]
    for n,end in [(36,60),(60,76),(76,88)]:
        metrics=[rp.stats_rr(rr[:bounds[n]],m['dates'][0],m['ends'][n-1]) for rr in returns[:fixed_count]]
        b=metrics[0]
        ids=[i for i,z in enumerate(metrics) if z['sharpe']>=b['sharpe']-.05 and z['max_drawdown']>=b['max_drawdown']-.03]
        win=max(ids,key=lambda i:(metrics[i]['cagr'],-i));part=returns[win][bounds[n]:bounds[end]]
        stitched.extend(part)
        wf.append(dict(training_end=m['ends'][n-1],test_start=m['dates'][n],test_end=m['ends'][end-1],
            chosen_id=win,spec=rows[win]['spec'],metrics=rp.stats_rr(part,m['dates'][n],m['ends'][end-1])))
    dump('TRAIN_TRIALS.json',rows)
    dump('WALKFORWARD.json',dict(folds=wf,metrics=rp.stats_rr(np.array(stitched),m['dates'][36],m['ends'][87]),
        baseline=rp.stats_rr(returns[0][bounds[36]:bounds[88]],m['dates'][36],m['ends'][87]),
        data_previously_researched=True))
    np.savez_compressed(P/'TRAIN_RETURNS.npz',returns=np.stack(returns),bounds=np.array(bounds))
    frozen=dict(created_at=datetime.now(timezone.utc).isoformat(),selected=selected,
        candidate_count=len(rows),predefined_count=fixed_count,admissible_count=len(good),
        primary='return_primary',baseline=base,code_sha256=sha(__file__),feature_sha256=sha(P/'FEATURES.npz'))
    dump('FROZEN.json',frozen);(P/'FROZEN.sha256').write_text(sha(P/'FROZEN.json')+'\n')
    for name,row in selected.items(): print('FROZEN',name,row,flush=True)
    print('Total',len(rows),'admissible',len(good),flush=True)


def evaluate():
    frozen=read(P/'FROZEN.json')
    assert sha(P/'FROZEN.json')==(P/'FROZEN.sha256').read_text().strip()
    assert frozen['code_sha256']==sha(__file__)
    assert frozen['feature_sha256']==sha(P/'FEATURES.npz')
    m,d,L,x,px,op,f=load();out={};allorders={}
    for name,row in dict(baseline=frozen['baseline'],**frozen['selected']).items():
        ch,sc=decisions(row['spec'],m,d,L,x,f,len(m['dates']));orders=rp.make_orders(ch,sc,m)
        met,eq,tr,hold=ex.e.engine.replay(px,op,orders,cost=0.,execution='close',
            start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
        blocks,dates,bounds,valid=rp.paths(px,m,len(ch));rr=rp.fast_returns(ch,blocks,valid)
        np.testing.assert_allclose(eq,np.r_[1.,np.cumprod(1+rr)]*100000.,rtol=1e-11,atol=1e-5)
        z=dict(metrics=met,slices=previous.cuts(eq,m),spec=row['spec'])
        assert abs(z['slices']['training2016_2023']['cagr']-row['training']['cagr'])<1e-10
        if name=='baseline':
            old=read(ex.P/'RESULTS.json')['return_primary']
            for key in ['cagr','sharpe','max_drawdown']:assert abs(met[key]-old['metrics'][key])<1e-10
            assert [o['symbols'] for o in orders]==[o['symbols'] for o in read(ex.P/'return_primary_ORDERS.json')]
        for label,mode,cost in [('next_open_zero','open',0.),('next_open_cost','open',.0005)]:
            mm,ee,_,_=ex.e.engine.replay(px,op,orders,cost=cost,execution=mode,
                start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
            z[label]=dict(metrics=mm,slices=previous.cuts(ee,m))
        dump(name+'_ORDERS.json',orders);dump(name+'_TRADES.json',tr);dump(name+'_HOLDINGS.json',hold)
        dump(name+'_DAILY.json',[dict(date=str(day.date()),equity=float(v)) for day,v in eq.items()])
        out[name]=z;allorders[name]=orders
        print(name,dict(full=met['cagr'],sharpe=met['sharpe'],early=z['slices']['early2016_2020']['cagr'],
            recent=z['slices']['original2021_2026']['cagr'],later=z['slices']['later34']['cagr']),flush=True)
    q=px.loc[m['dates'][0]:m['ends'][-1],'QQQ'];qe=q/q.iloc[0]*100000.
    out['qqq']=dict(metrics=ex.e.engine.stats(qe),slices=previous.cuts(qe,m))
    dump('RESULTS.json',out)
    comparison=[]
    for i,day in enumerate(m['dates']):
        r=dict(date=day,end=m['ends'][i],split='early' if i<53 else 'recent_train' if i<88 else 'later34')
        for name,orders in allorders.items():
            ss=orders[i]['symbols'];r[name]=dict(symbols=ss,monthly_return=float((px.loc[m['ends'][i],ss]/px.loc[day,ss]-1).mean()) if ss else 0.)
        comparison.append(r)
    dump('MONTH_COMPARISON.json',comparison)
    changes=[p for p,h in read(P/'PROTECTED.json').items() if sha(p)!=h]
    assert not changes,changes
    dump('VALIDATION.json',dict(baseline_holdings_and_metrics_exact=True,independent_path_parity=True,
        training_freeze_replay_parity=True,protected_files=len(read(P/'PROTECTED.json')),
        protected_changes=changes,no_author_targets=True,pristine_oos=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['initialize','prepare','train','evaluate'])
    globals()[ap.parse_args().phase]()
