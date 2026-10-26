"""Frozen monthly session, asset-type and actual-month pairwise experiments."""
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
sys.path.insert(0, str(O / 'monthly_daily_episodes'))
import episodes as ep
sys.path.insert(0, str(O / 'monthly_architecture_restart'))
spec = importlib.util.spec_from_file_location('session_restart_inputs', O / 'monthly_architecture_restart/study.py')
arch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arch)
rp, ex, read, sha = ep.rp, ep.ex, ep.read, ep.sha
TRAIN = 88
BASES = ['original65', 'simple']
SESSION = ['session_log_spread', 'overnight_sign_mean', 'intraday_sign_mean']
TYPE = ['plain_pooled', 'plain_partial_10', 'plain_partial_100', 'plain_independent']
PAIR = ['plain_pairwise', 'state_pairwise', 'state_pooled']


def dump(name, value):
    (P / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda v: v.item() if isinstance(v, np.generic) else v) + '\n')


def grid():
    rows = []
    for base in BASES:
        rows.append(dict(id=len(rows), base=base, family='control', key='none', budget=0., direction=1))
        for key, direction, budget in itertools.product(SESSION, [-1, 1], [.025, .075]):
            rows.append(dict(id=len(rows), base=base, family='session', key=key, budget=budget, direction=direction))
        for family, keys in [('type', TYPE), ('pair', PAIR)]:
            for key, budget in itertools.product(keys, [.025, .075]):
                rows.append(dict(id=len(rows), base=base, family=family, key=key, budget=budget, direction=1))
    return rows


def initialize():
    assert not (P / 'PLAN.json').exists()
    dump('PLAN.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        purpose='Test long-horizon overnight/intraday composition, partially pooled stock/ETF Ridge and actual-month state-conditioned pairwise return ranking.',
        fixed='122 existing Top32-estimate plus assumed theme17 pools and eligibility;2016-08-17 initialization then monthends,valuation through2026-09-24.3equal/2half/<2cash,long-only,no leverage,fixed monthly adjusted units.No author labels or new securities.',
        bases='Frozen65.34 score with original core50/52%; independently75%raw210daymomentum ending42sessionsago+25%vol126. All candidates compete for all3slots;66.36 boundary-locked rule is preserved only as an exact control.',
        session='Adjustedopen=rawopen*adjclose/rawclose. Decompose log(adjopen/previousadjclose) and log(adjclose/adjopen),rolling210 ending42sessionsago. Three signals:overnight-minus-intraday sum,overnight sign mean,intraday sign mean;each positive/negative and2.5/7.5%bounded correction. Identity/prefix/split tests;missing component gives zero change,never future eligibility filtering.',
        type_models='Annual trailing5years,priorDec31 matured next-calendar-month labels,month balanced with24month decay,min18months,alpha10. Shared Ridge vs type deviations10x/100x penalty vs independent-type diagnostic. Plain10 features:rawmomentum,recent-risk momentum,volatility,own-sector relative,and six price-volume ranks. No ticker/date identity features or fixed asset-type allocations.',
        pair_models='Same actual-month centered excessSPY returns;allwithinmonth unordered pairs,2x weight if crossing PAST simple-score top3,otherwise1. Plain10 features vs28feature pack adding six auxiliary×breadth/trend/highvol interactions. Pooled-state prediction is paired control. No realized-top3 weights or author selections.',
        labels='Reuse architecture_restart actualcalendar-month centered clipped excessSPY returns. Future-complete mask is training-label validity ONLY; event eligibility exactly original. Prior2016 training membership remains inherited proxy.',
        correction='Effect=session centeredpercentile or0.5*tanh(prediction/.05). Let support=(base-core)/(1-coreweight); replace budget of support with clip(support+effect,-.5,.5). Thus zero effect/unavailable is exact nochange,core preserved,total weights1,and no percentile amplification of tiny ML forecasts.',
        grid='54 configurations fixed before returns:2 bases x(1control+12session+8type+6pair). No additive combinations or grid extensions after evaluation.',
        selection='First88months through2023-11-30 only. Relative original65:trainCAGR>=base,Sharpe>=base-.05,DD>=base-.03,early53CAGR>=base,recent35CAGR>=base-.02. Highest admitted trainCAGR primary;fallback original65. Freeze per-family training-best on each base as descriptive diagnostics whether or not admitted, and matched pooled/type/pair ablations at their selected budget. Never promote on later34 results.',
        execution='Zero-cost same-final-close idealized primary. Identical frozen plans at nextopen0cost/5bps-per-side sensitivities. Daily fixed-unit engine vs independent paths; partial first/last years explicitly marked.',
        reporting='All training trials,admission,early/recent/later sections,changedmonths/top2/stock contributions,undo mostfavorable1and3changes as diagnosis only,within-training chronological selection and zero-change placebo. No untouched-OOS claim;history has been repeatedly researched.',
        protection='No original prices,ledgers,rules,plans,automation,fingerprints or Git mutations. New files only here. No guarantees of improvement or author recovery.'))
    dump('GRID.json', grid())
    protect = read(O / 'monthly_architecture_restart/PROTECTED.json')
    for folder in ['monthly_architecture_restart', 'monthly_daily_episodes', 'paired_stock_monitor']:
        protect.update({str(f): sha(f) for f in (O / folder).rglob('*')
                        if f.is_file() and f.suffix != '.pyc' and f.name != 'run.lock'})
    dump('PROTECTED.json', protect)


def prepare():
    from session_features import build_session_features
    m, d, px, op, a = ep.load()
    am = read(arch.P / 'META.json')
    ad = dict(np.load(arch.P / 'DATA.npz'))
    np.testing.assert_array_equal(ad['eligible'], d['eligible'])
    assert m['symbols'] == am['monthly']['symbols'] and m['dates'] == am['monthly']['dates']
    panel, info = arch.old.panel_load(m['etfs'])
    frames, audit = build_session_features(panel)
    arrays = {k: ad[k] for k in ['dates', 'ends', 'event_dates', 'offsets', 'y', 'train_good', 'eligible']}
    arrays.update(base_original65=a['baseline'], core_original65=a['core'], cw_original65=a['core_weight'],
                  base_simple=ad['event_all_raw_vol'], core_simple=.75 * ad['event_all_raw'],
                  cw_simple=np.full((122, 1), .75))
    for key in SESSION:
        raw = np.stack([frames[key].loc[day, ss].to_numpy() for day, ss in zip(m['dates'], m['symbols'])])
        arrays[key] = np.stack([arch.rank_values(row) for row in raw])
        arrays['valid_' + key] = np.isfinite(raw)
    for prefix, count in [('train', len(ad['dates'])), ('event', len(ad['event_dates']))]:
        core = np.column_stack([ad[prefix + '_all_' + key].reshape(-1)
                                for key in ['raw', 'recent_risk', 'vol', 'sector']])
        aux = ad['aux_all' if prefix == 'train' else 'event_aux_all']
        arrays[prefix + '_plain'] = np.column_stack([core, aux[:, :6]])
        arrays[prefix + '_state'] = np.column_stack([core, aux])
        assert arrays[prefix + '_plain'].shape == (count, 10)
        assert arrays[prefix + '_state'].shape == (count, 28)
    etfs = set(m['etfs'])
    arrays['train_type'] = np.array([int(s in etfs) for ss in am['training']['symbols'] for s in ss])
    arrays['event_type'] = np.array([int(s in etfs) for ss in m['symbols'] for s in ss])
    arrays['boundary_score'] = ad['train_all_raw_vol']
    past_good = (panel['adj'].notna().rolling(295).sum().eq(295)
                 & panel['adj'].gt(0) & (panel['adj'].shift(42)/panel['adj'].shift(252)-1).gt(0))
    boundary = np.zeros(len(ad['dates']), dtype=bool)
    for i, (day, ss) in enumerate(zip(am['training']['dates'], am['training']['symbols'])):
        lo, hi = ad['offsets'][i:i+2]
        good = past_good.loc[day, ss].to_numpy()
        ids = np.flatnonzero(good)
        order = ids[np.argsort(-arrays['boundary_score'][lo:hi][ids], kind='stable')]
        boundary[lo+order[:3]] = True
    arrays['boundary_top3'] = boundary
    ch = rp.choose(arrays['base_original65'], arrays['eligible'], m['symbols'], 3)
    orders = rp.make_orders(ch, arrays['base_original65'], m)
    assert orders == read(ep.P / 'original65_ORDERS.json')
    for s in px:
        np.testing.assert_allclose(panel['adj'][s].reindex(px.index), px[s], equal_nan=True, rtol=1e-12, atol=1e-8)
    np.savez_compressed(P / 'DATA.npz', **arrays)
    dump('META.json', am)
    hashes = info['source_hashes']
    for f in [arch.P/'DATA.npz', arch.P/'META.json', ep.P/'FROZEN.json', ep.P/'FORECASTS.npz',
              ep.P/'primary_ORDERS.json', ep.P/'original65_ORDERS.json', Path(ep.__file__), Path(rp.__file__), Path(ex.e.engine.__file__)]:
        hashes[str(f)] = sha(f)
    dump('INPUT_HASHES.json', hashes)
    dump('PREPARE_VALIDATION.json', dict(session=audit, original65_orders_exact=True,
        old_prices_exact=True, trading_eligibility_unchanged=True, training_rows=len(ad['dates']),
        completed_labels=int(ad['train_good'].sum()), session_missing_eligible={
            k:int((arrays['eligible'] & ~arrays['valid_'+k]).sum()) for k in SESSION}))
    print('Prepared', len(ad['dates']), 'rows and122 unchanged trading events', flush=True)


def forecasts():
    from models import annual_forecasts
    a = dict(np.load(P / 'DATA.npz'))
    out, audit = {}, {}
    for pack in ['plain', 'state']:
        r = annual_forecasts(a['dates'], a['ends'], a['train_'+pack], a['y'], a['train_good'],
            a['train_type'], a['event_dates'], a['event_'+pack], a['eligible'].reshape(-1),
            a['event_type'], alpha=10., half_life_months=24., min_months=18,
            deviation_strengths=(10.,100.), boundary_top3=a['boundary_top3'])
        for key in ['pooled', 'partial_10', 'partial_100', 'independent', 'pairwise']:
            out[pack+'_'+key] = r[key].reshape(122,49)
            out['valid_'+pack+'_'+key] = r['available'][key].reshape(122,49)
        audit[pack] = r['audit']
        print('Forecasts', pack, flush=True)
    np.savez_compressed(P / 'FORECASTS.npz', **out)
    dump('FORECAST_AUDIT.json', audit)
    dump('FORECAST_INPUT_HASHES.json', {f:sha(P/f) for f in ['DATA.npz','models.py']})


def load():
    for file, h in read(P / 'INPUT_HASHES.json').items():
        assert sha(file) == h, file
    for file,h in read(P/'FORECAST_INPUT_HASHES.json').items():
        assert sha(P/file)==h,file
    m, _, px, op, _ = ep.load()
    return m, {**dict(np.load(P/'DATA.npz')), **dict(np.load(P/'FORECASTS.npz'))}, px, op


def score(row, a, n):
    base = a['base_'+row['base']][:n]
    if row['budget'] == 0:
        return base.copy()
    core, cw = a['core_'+row['base']][:n], a['cw_'+row['base']][:n]
    support = (base-core)/(1-cw)
    values = a[row['key']][:n]
    effect = values if row['family'] == 'session' else .5*np.tanh(values/.05)
    effect = np.where(a['valid_'+row['key']][:n], row['direction']*effect, 0.)
    delta = np.where(effect == 0., 0., np.clip(support+effect, -.5, .5)-support)
    return base + row['budget']*delta


def choose(row,a,m,n):
    sc=score(row,a,n)
    return rp.choose(sc,a['eligible'][:n],m['symbols'][:n],3),sc


def measure(rr,m,bounds,n=TRAIN):
    split=53 if n>=88 else n//2
    return dict(training=rp.stats_rr(rr[:bounds[n]],m['dates'][0],m['ends'][n-1]),
                early=rp.stats_rr(rr[:bounds[split]],m['dates'][0],m['ends'][split-1]),
                recent=rp.stats_rr(rr[bounds[split]:bounds[n]],m['dates'][split],m['ends'][n-1]))


def admitted(z,b):
    return (z['training']['cagr'] >= b['training']['cagr']-1e-12
        and z['training']['sharpe'] >= b['training']['sharpe']-.05
        and z['training']['max_drawdown'] >= b['training']['max_drawdown']-.03
        and z['early']['cagr'] >= b['early']['cagr']-1e-12
        and z['recent']['cagr'] >= b['recent']['cagr']-.02)


def train():
    assert not (P/'FROZEN.json').exists()
    m,a,px,_=load()
    blocks,_,bounds,valid=rp.paths(px,m,TRAIN)
    rows,returns=[],[]
    for spec in grid():
        ch,_=choose(spec,a,m,TRAIN)
        rr=rp.fast_returns(ch,blocks,valid)
        rows.append(dict(spec=spec,**measure(rr,m,bounds)))
        returns.append(rr)
    b=rows[0]
    for row in rows:row['admitted']=admitted(row,b)
    new=[z for z in rows if z['admitted'] and z['spec']['family']!='control']
    best=lambda zs:max(zs,key=lambda z:(z['training']['cagr'],-z['spec']['id']))
    selected=dict(primary=best([b]+new),original65=b,simple=next(z for z in rows if z['spec']['base']=='simple' and z['spec']['family']=='control'))
    for base,family in itertools.product(BASES,['session','type','pair']):
        selected[base+'_'+family]=best([z for z in rows if z['spec']['base']==base and z['spec']['family']==family])
    # Budget-matched, training-frozen diagnostics; not selected on later results.
    for family,keys in [('type',TYPE),('pair',['plain_pooled']+PAIR)]:
        r=selected['original65_'+family]['spec']
        for key in keys:
            selected['matched_'+family+'_'+key]=next(z for z in rows if z['spec']['base']=='original65'
                and z['spec']['key']==key and z['spec']['budget']==r['budget'])
    wf,parts,baseparts=[],[],[]
    for n,end in [(36,60),(60,76),(76,88)]:
        ms=[dict(spec=z['spec'],**measure(rr,m,bounds,n)) for z,rr in zip(rows,returns)]
        win=best([z for z in ms if admitted(z,ms[0])])
        i=win['spec']['id'];piece=returns[i][bounds[n]:bounds[end]]
        parts.extend(piece);baseparts.extend(returns[0][bounds[n]:bounds[end]])
        wf.append(dict(train_end=m['ends'][n-1],test_end=m['ends'][end-1],spec=win['spec']))
    dump('WALKFORWARD.json',dict(folds=wf,metrics=rp.stats_rr(np.array(parts),m['dates'][36],m['ends'][87]),baseline=rp.stats_rr(np.array(baseparts),m['dates'][36],m['ends'][87]),pristine_oos=False))
    dump('TRAIN_TRIALS.json',rows)
    files=['PLAN.json','GRID.json','DATA.npz','META.json','INPUT_HASHES.json','FORECASTS.npz',
           'FORECAST_AUDIT.json','FORECAST_INPUT_HASHES.json','study.py','models.py','session_features.py']
    dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),selected=selected,
        new_admitted=len(new),trials=len(rows),hashes={f:sha(P/f) for f in files},
        diagnostics_may_be_unqualified=True,author_labels_used=False))
    (P/'FROZEN.sha256').write_text(sha(P/'FROZEN.json')+'\n')
    print('Frozen',len(new),'admitted new configs;primary',selected['primary']['spec'],flush=True)


def evaluate():
    fr=read(P/'FROZEN.json')
    assert sha(P/'FROZEN.json')==(P/'FROZEN.sha256').read_text().strip()
    for f,h in fr['hashes'].items():assert sha(P/f)==h,f
    m,a,px,op=load()
    blocks,_,bounds,valid=rp.paths(px,m,122)
    allorders={}
    for name,z in fr['selected'].items():
        ch,sc=choose(z['spec'],a,m,122);allorders[name]=rp.make_orders(ch,sc,m)
    allorders['previous66']=read(ep.P/'primary_ORDERS.json')
    allorders['qqq']=[dict(date=m['dates'][0],signal_date=m['dates'][0],end=m['ends'][-1],symbols=['QQQ'],scores=[])]
    results={}
    for name,orders in allorders.items():
        met,eq,tr,hold=ex.e.engine.replay(px,op,orders,cost=0.,execution='close',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
        # The first year is also partial, regardless of the inherited engine flag.
        for row in met['annual_returns']:
            if row['year'] in [2016,2026]:row['partial']=True
        slices=arch.old.prior.old.previous.cuts(eq,m)
        if name!='qqq':
            ch=[[m['symbols'][i].index(s) for s in order['symbols']] for i,order in enumerate(orders)]
            rr=rp.fast_returns(ch,blocks,valid)
            np.testing.assert_allclose(eq,np.r_[1.,np.cumprod(1+rr)]*100000,rtol=1e-11,atol=1e-5)
        if name in fr['selected']:
            assert abs(slices['training2016_2023']['cagr']-fr['selected'][name]['training']['cagr'])<1e-10
        results[name]=dict(metrics=met,slices=slices,selection=fr['selected'].get(name))
        for label,cost in [('next_open_zero',0.),('next_open_cost',.0005)]:
            mm,ee,_,_=ex.e.engine.replay(px,op,orders,cost=cost,execution='open',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
            for row in mm['annual_returns']:
                if row['year'] in [2016,2026]:row['partial']=True
            results[name][label]=dict(metrics=mm,slices=arch.old.prior.old.previous.cuts(ee,m))
        for suffix,data in [('ORDERS',orders),('DAILY',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()]),('TRADES',tr),('HOLDINGS',hold)]:dump(name+'_'+suffix+'.json',data)
        print(name,round(met['cagr']*100,3),round(met['sharpe'],3),flush=True)
    for name,reference in [('original65','original65'),('previous66','primary')]:
        ref=read(ep.P/'RESULTS.json')[reference]['metrics']
        for k in ['cagr','sharpe','max_drawdown']:assert abs(results[name]['metrics'][k]-ref[k])<1e-10
    comparison=[]
    for i,day in enumerate(m['dates']):
        row=dict(date=day,end=m['ends'][i],split='early53' if i<53 else 'recent35' if i<88 else 'later34')
        for name,orders in allorders.items():
            if name=='qqq':continue
            ss=orders[i]['symbols']
            ret=px.loc[m['ends'][i],ss]/px.loc[day,ss]-1 if ss else pd.Series(dtype=float)
            row[name]=dict(symbols=ss,monthly_return=float(ret.mean()) if ss else 0.,stock_contributions={s:float(v/len(ss)) for s,v in ret.items()})
        comparison.append(row)
    dump('MONTH_COMPARISON.json',comparison)
    attribution={}
    for name in fr['selected']:
        if name in ['original65','simple']:continue
        base=fr['selected'][name]['spec']['base']
        changes=[]
        for row in comparison:
            x,b=row[name],row[base]
            if set(x['symbols'])!=set(b['symbols']):
                changes.append(dict(date=row['date'],end=row['end'],split=row['split'],old=b['symbols'],new=x['symbols'],
                    difference=x['monthly_return']-b['monthly_return'],
                    log_contribution=float(np.log1p(x['monthly_return'])-np.log1p(b['monthly_return'])),
                    top2_lost=len(set(b['symbols'][:2])-set(x['symbols']))))
        best=sorted([z for z in changes if z['log_contribution']>0],key=lambda z:-z['log_contribution'])
        years=results[name]['metrics']['years'];wealth=1+results[name]['metrics']['total_return']
        attribution[name]=dict(base=base,changed_months=len(changes),changes=changes,
            top1_favorable_log=sum(z['log_contribution'] for z in best[:1]),top3_favorable_log=sum(z['log_contribution'] for z in best[:3]),
            undo_best1_diagnostic_cagr=float((wealth*np.exp(-sum(z['log_contribution'] for z in best[:1])))**(1/years)-1),
            undo_best3_diagnostic_cagr=float((wealth*np.exp(-sum(z['log_contribution'] for z in best[:3])))**(1/years)-1))
    dump('ATTRIBUTION.json',attribution)
    changes=[p for p,h in read(P/'PROTECTED.json').items() if sha(p)!=h]
    assert not changes,changes
    dump('RESULTS.json',results)
    dump('VALIDATION.json',dict(old_controls_exact=True,independent_daily_fixed_unit_replay=True,
        training_freeze_parity=True,protected_files=len(read(P/'PROTECTED.json')),protected_changes=changes,
        new_observation_accounts=False,pristine_oos=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=['initialize','prepare','forecasts','train','evaluate'])
    globals()[ap.parse_args().phase]()
