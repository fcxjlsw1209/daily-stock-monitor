"""Predeclared monthly post-earnings-drift experiment; research outputs only."""
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
spec = importlib.util.spec_from_file_location('earnings_existing_inputs', O/'monthly_core_expansion/study.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
session, ep, arch, rp, ex, read, sha = (previous.session, previous.ep, previous.arch,
    previous.rp, previous.ex, previous.read, previous.sha)
REPAIR = O/'monthly_earnings_repair'
TRAIN = 88
BASES = ['original65', 'simple']


def dump(name, value):
    (P/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda v: v.item() if isinstance(v, np.generic) else v)+'\n')


def grid():
    rows = []
    for base in BASES:
        rows.append(dict(id=len(rows), base=base, family='control', key='none', budget=0.))
        for key, budget in itertools.product(['event10', 'event21', 'matched_recent10',
                'matched_recent21', 'ridge_plain', 'ridge_earn'], [.025, .05]):
            family = 'ridge' if key.startswith('ridge') else 'event' if key.startswith('event') else 'recent'
            rows.append(dict(id=len(rows), base=base, family=family, key=key, budget=budget))
    return rows


def initialize():
    assert not (P/'PLAN.json').exists(), 'Do not overwrite preregistration'
    dump('PLAN.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        purpose='First earnings-factor experiment after candidate-scoped data admission. Test incremental information, not exact author reproduction.',
        data='Same122 estimated historical Top32+assumedETF17 pools, 2016-08-17 through2026-09-24. Training180months2011..2025; fiscal dates/publication evidence verified separately. No new universe, no author holding labels.',
        strategy='Long momentum skips latest42sessions; existing positive longtrend/295price eligibility. Three equal positions, two half positions, fewer than two cash. Monthly fixed adjusted-return units, no leverage, no intramonth exits or pullback requirement.',
        controls='Exact original65.34 score; independent75%rawlongmomentum+25%vol126 score. Previous66.36 frozen orders are incumbent comparison and fallback, not retuned.',
        factors='Use latest verified event, conservative known-by close, completed10/21XNYS-session stock return minus same-endpoint SPY return. Event features remain missing for ETF, pre-first-release and immature windows. No older-event fallback, no EPS surprise or initial announcement gap claims.',
        manual='Positive direction .5*tanh(market_adjusted_return/.05), budgets2.5/5%. Matched recent10/21 simple return minus SPY ending at signal, exact same event availability mask. Controls distinguish earnings anchoring from ordinary recent momentum. No raw cross-company volume comparison.',
        model='Paired plain10 versus plain10+event10+event21+(event21-event10) Ridge. Same jointly finite eligible training rows, same prediction availability. Annual prior-Dec31 cutoff, previous5years signal dates, all labels matured, alpha10,24month half-life, equal aggregate month weight,minimum18months. Training-only standardization; missing forecasts remainNaN, only score correction is zero.',
        target='Reused validated next-calendar-month clipped excessSPY centered across historical eligible names. Full future path validity gates training labels only, never trading eligibility. No ticker/year/author identity features.',
        correction='No core change. support=(base-core)/(1-coreweight); score=base+budget*(clip(support+effect,-.5,.5)-support). Missing or exactly zero effect makes exactly zero adjustment. Ridge effect=.5*tanh(pred/.05). All three slots compete; no percentile amplification.',
        grid='26 fixed configurations:2bases*(1control+4manualsignals*2budgets+2Ridgearms*2budgets). No new variants after later-period evaluation.',
        selection='First88 periods only, ending2023-11-30. Candidate must beat incumbent66 trainingCAGR, Sharpe no worse by.05, DD no worse by.03, early53CAGR not lower, recent35CAGR at most.02 lower. Highest qualified earnings-arm trainingCAGR primary; otherwise exact incumbent66. Freeze per-base best event+same horizon/budget recent control, best earningsRidge+samebudget plain control, even if unqualified. Later34 only after freeze. No promotion based on later performance.',
        execution='Same final close signal/fill and zero cost idealized primary, matching previous research. Separately replay identical frozen orders next-openzero and next-open5bp/side. Full-close information cannot guarantee same-close execution. Partial first/last periods disclosed.',
        robustness='Training prefix36/60/76 selection diagnostics; paired event-vs-recent and earn-vs-plain monthly contributions; undo most profitable1/3 changes as diagnostic only. Independent fixed-unit paths vs engine; old65/66 parity; read-only source hashes and account protection.',
        limitations='Previously researched history, not pristineOOS. Historical pool/cap/sector proxies and uncertain authorETFlist remain. Current source archive verification is not immutable contemporaneous news feed. No automatic observer/ledger/plan changes.'))
    dump('GRID.json', grid())
    paths = set(read(previous.P/'PROTECTED.json'))
    for folder in list(O.glob('*_monitor'))+[previous.P, REPAIR]:
        for f in folder.rglob('*'):
            if f.is_file() and f.suffix != '.pyc' and f.name != 'run.lock':
                paths.add(str(f))
    dump('PROTECTED.json', {str(f): sha(f) for f in sorted(paths) if Path(f).is_file()})


def assert_hashes(mapping, root=None):
    for file, digest in mapping.items():
        path = Path(file) if Path(file).is_absolute() or root is None else root/file
        assert sha(path) == digest, str(path)


def prepare():
    sys.path.insert(0, str(P))
    from earnings_features import build_features
    gate = read(REPAIR/'SCOPED_READINESS.json')
    assert gate['scoped_post_known_drift_data_ready'] and gate['unresolved_count'] == 0
    assert_hashes(gate['inputs'], REPAIR)
    m, old, px, op = previous.load()
    am = read(arch.P/'META.json')
    tm = am['training']
    assert m['dates'] == am['monthly']['dates'] and m['symbols'] == am['monthly']['symbols']
    panel, info = arch.old.panel_load(m['etfs'])
    a = {k: old[k] for k in ['eligible', 'dates', 'ends', 'event_dates', 'offsets',
        'train_plain', 'event_plain', 'train_good', 'train_eligible', 'target_return']}
    for base in BASES:
        for prefix in ['base_', 'core_', 'cw_']:
            a[prefix+base] = old[prefix+base]
    audits = {}
    for scope, meta in [('monthly', m), ('training', tm)]:
        doc = read(REPAIR/(scope.upper()+'_FEATURES.json'))
        features, audits[scope] = build_features(doc, meta, panel['adj'], set(m['etfs']))
        for key, value in features.items():
            a[scope+'_'+key] = value
    assert np.isfinite(a['target_return'][a['train_good']]).all()
    assert rp.make_orders(rp.choose(a['base_original65'], a['eligible'], m['symbols'], 3),
        a['base_original65'], m) == read(ep.P/'original65_ORDERS.json')
    np.savez_compressed(P/'DATA.npz', **a)
    dump('META.json', dict(monthly=m, training=tm))
    dump('FEATURE_AUDIT.json', audits)
    hashes = dict(info['source_hashes'])
    paths = [REPAIR/'SCOPED_READINESS.json', REPAIR/'EVENTS.json',
        REPAIR/'MONTHLY_FEATURES.json', REPAIR/'TRAINING_FEATURES.json',
        previous.P/'DATA.npz', previous.P/'INPUT_HASHES.json',
        previous.P/'TARGET_AUDIT.json', arch.P/'META.json',
        ep.P/'FROZEN.json', ep.P/'primary_ORDERS.json', ep.P/'original65_ORDERS.json',
        Path(previous.__file__), Path(session.__file__), Path(arch.__file__),
        Path(rp.__file__), Path(ex.e.engine.__file__), P/'earnings_features.py',
        REPAIR/'research_calendar.py', O/'monthly_architecture_restart/residual_model.py']
    for f in paths:
        hashes[str(f)] = sha(f)
    for key, digest in gate['inputs'].items():
        f = Path(key) if Path(key).is_absolute() else REPAIR/key
        hashes[str(f)] = digest
    dump('INPUT_HASHES.json', hashes)
    dump('PREPARE_VALIDATION.json', dict(original65_orders_exact=True,
        eligibility_unchanged=True, monthly_periods=len(m['dates']), training_months=len(tm['dates']),
        ready_training_ml=int((a['train_good'] & a['training_valid_ml'].reshape(-1)).sum()),
        ready_event_ml=int((a['eligible'] & a['monthly_valid_ml']).sum())))
    print('Prepared admitted earnings features and unchanged baseline', flush=True)


def forecasts():
    assert_hashes(read(P/'INPUT_HASHES.json'))
    sys.path.insert(0, str(P))
    from earnings_model import annual_forecasts
    a = dict(np.load(P/'DATA.npz'))
    result = annual_forecasts(a['dates'], a['ends'], a['train_plain'],
        a['training_ml_features'].reshape(-1, 3), a['target_return'],
        a['train_good'] & a['training_valid_ml'].reshape(-1), a['event_dates'],
        a['event_plain'], a['monthly_ml_features'].reshape(-1, 3),
        a['eligible'].reshape(-1) & a['monthly_valid_ml'].reshape(-1))
    shape = a['eligible'].shape
    np.savez_compressed(P/'FORECASTS.npz',
        **{'ridge_'+key: result[key].reshape(shape) for key in ['plain', 'earn', 'available']})
    dump('FORECAST_AUDIT.json', result['audit'])
    dump('FORECAST_INPUT_HASHES.json', {f: sha(P/f) for f in ['DATA.npz', 'earnings_model.py']})
    print('Fitted annual paired Ridge forecasts', flush=True)


def load():
    assert_hashes(read(P/'INPUT_HASHES.json'))
    assert_hashes(read(P/'FORECAST_INPUT_HASHES.json'), P)
    m, _, px, op = previous.load()
    return m, {**dict(np.load(P/'DATA.npz')), **dict(np.load(P/'FORECASTS.npz'))}, px, op


def corrected_score(row, a, n):
    b = row['base']
    base = a['base_'+b][:n].copy()
    if row['family'] == 'control':
        return base
    if row['family'] == 'ridge':
        values, valid = a[row['key']][:n], a['ridge_available'][:n]
    else:
        values = a['monthly_'+row['key']][:n]
        valid = a['monthly_valid'+('10' if row['key'].endswith('10') else '21')][:n]
    assert np.isfinite(values[valid]).all()
    effect = np.zeros(values.shape)
    effect[valid] = .5*np.tanh(values[valid]/.05)
    support = (base-a['core_'+b][:n])/(1-a['cw_'+b][:n])
    delta = np.where(effect == 0, 0., np.clip(support+effect, -.5, .5)-support)
    return base+row['budget']*delta


def choices(row, a, m, n):
    score = corrected_score(row, a, n)
    return rp.choose(score, a['eligible'][:n], m['symbols'][:n], 3), score


def incumbent_choices(m):
    orders = read(ep.P/'primary_ORDERS.json')
    return [[m['symbols'][i].index(s) for s in order['symbols']] for i, order in enumerate(orders)]


def train():
    assert not (P/'FROZEN.json').exists(), 'Never overwrite frozen selection'
    m, a, px, _ = load()
    blocks, _, bounds, valid = rp.paths(px, m, TRAIN)
    incumbent_rr = rp.fast_returns(incumbent_choices(m)[:TRAIN], blocks, valid)
    incumbent = session.measure(incumbent_rr, m, bounds)
    rows, returns = [], []
    for row in grid():
        ch, _ = choices(row, a, m, TRAIN)
        rr = rp.fast_returns(ch, blocks, valid)
        result = dict(spec=row, **session.measure(rr, m, bounds))
        result['admitted'] = session.admitted(result, incumbent)
        rows.append(result)
        returns.append(rr)
    best = lambda xs: max(xs, key=lambda z: (z['training']['cagr'], -z['spec']['id']))
    find = lambda **kw: next(z for z in rows if all(z['spec'][k] == v for k, v in kw.items()))
    is_earn = lambda z: z['spec']['family'] == 'event' or z['spec']['key'] == 'ridge_earn'
    admitted = [z for z in rows if is_earn(z) and z['admitted']]
    selected = {b: find(base=b, family='control') for b in BASES}
    if admitted:
        selected['primary'] = best(admitted)
    for b in BASES:
        z = best([z for z in rows if z['spec']['base'] == b and z['spec']['family'] == 'event'])
        selected[b+'_event'] = z
        selected[b+'_matched_recent'] = find(base=b,
            key=z['spec']['key'].replace('event', 'matched_recent'), budget=z['spec']['budget'])
        z = best([z for z in rows if z['spec']['base'] == b and z['spec']['key'] == 'ridge_earn'])
        selected[b+'_ridge_earn'] = z
        selected[b+'_ridge_plain'] = find(base=b, key='ridge_plain', budget=z['spec']['budget'])
    folds, parts, control = [], [], []
    for n, end in [(36, 60), (60, 76), (76, 88)]:
        incumbent_prefix = session.measure(incumbent_rr, m, bounds, n)
        prefix = [dict(spec=z['spec'], **session.measure(rr, m, bounds, n))
            for z, rr in zip(rows, returns)]
        permitted = [z for z in prefix if is_earn(z) and session.admitted(z, incumbent_prefix)]
        win = best(permitted) if permitted else None
        rr = returns[win['spec']['id']] if win else incumbent_rr
        parts.extend(rr[bounds[n]:bounds[end]])
        control.extend(incumbent_rr[bounds[n]:bounds[end]])
        folds.append(dict(train_end=m['ends'][n-1], test_end=m['ends'][end-1],
            spec=win['spec'] if win else 'incumbent66'))
    dump('WALKFORWARD.json', dict(folds=folds,
        metrics=rp.stats_rr(np.array(parts), m['dates'][36], m['ends'][87]),
        incumbent=rp.stats_rr(np.array(control), m['dates'][36], m['ends'][87]), pristine_oos=False))
    dump('TRAIN_TRIALS.json', rows)
    files = ['PLAN.json', 'GRID.json', 'DATA.npz', 'META.json', 'INPUT_HASHES.json',
        'FEATURE_AUDIT.json', 'FORECASTS.npz', 'FORECAST_AUDIT.json', 'FORECAST_INPUT_HASHES.json',
        'study.py', 'earnings_features.py', 'earnings_model.py']
    dump('FROZEN.json', dict(created_at=datetime.now(timezone.utc).isoformat(), selected=selected,
        primary_source='candidate' if admitted else 'incumbent66', new_admitted=len(admitted),
        incumbent_training=incumbent, trials=len(rows), hashes={f: sha(P/f) for f in files},
        diagnostics_may_be_unqualified=True, author_labels_used=False))
    (P/'FROZEN.sha256').write_text(sha(P/'FROZEN.json')+'\n')
    print('Frozen selection; new admitted:', len(admitted), flush=True)


def evaluate():
    frozen = read(P/'FROZEN.json')
    assert sha(P/'FROZEN.json') == (P/'FROZEN.sha256').read_text().strip()
    assert_hashes(frozen['hashes'], P)
    m, a, px, op = load()
    blocks, _, bounds, valid = rp.paths(px, m, len(m['dates']))
    orders = {}
    for name, row in frozen['selected'].items():
        ch, score = choices(row['spec'], a, m, len(m['dates']))
        orders[name] = rp.make_orders(ch, score, m)
    orders['previous66'] = read(ep.P/'primary_ORDERS.json')
    if frozen['primary_source'] == 'incumbent66':
        orders['primary'] = orders['previous66']
    orders['qqq'] = [dict(date=m['dates'][0], signal_date=m['dates'][0], end=m['ends'][-1], symbols=['QQQ'], scores=[])]
    results, memo = {}, {}
    for name, oo in orders.items():
        key = json.dumps(oo, sort_keys=True)
        if key not in memo:
            met, eq, tr, hold = ex.e.engine.replay(px, op, oo, cost=0., execution='close',
                start=pd.Timestamp(m['dates'][0]), end=pd.Timestamp(m['ends'][-1]))
            for row in met['annual_returns']:
                if row['year'] in [2016, 2026]: row['partial'] = True
            common = dict(metrics=met, slices=arch.old.prior.old.previous.cuts(eq, m))
            for label, cost in [('next_open_zero', 0.), ('next_open_cost', .0005)]:
                mm, ee, _, _ = ex.e.engine.replay(px, op, oo, cost=cost, execution='open',
                    start=pd.Timestamp(m['dates'][0]), end=pd.Timestamp(m['ends'][-1]))
                for row in mm['annual_returns']:
                    if row['year'] in [2016, 2026]: row['partial'] = True
                common[label] = dict(metrics=mm, slices=arch.old.prior.old.previous.cuts(ee, m))
            memo[key] = common, eq, tr, hold
        common, eq, tr, hold = memo[key]
        if name != 'qqq':
            ch = [[m['symbols'][i].index(s) for s in row['symbols']] for i, row in enumerate(oo)]
            rr = rp.fast_returns(ch, blocks, valid)
            np.testing.assert_allclose(eq, np.r_[1., np.cumprod(1+rr)]*100000, rtol=1e-11, atol=1e-5)
        if name in frozen['selected']:
            assert abs(common['slices']['training2016_2023']['cagr']-frozen['selected'][name]['training']['cagr']) < 1e-10
        results[name] = dict(common, selection=frozen['selected'].get(name))
        for suffix, data in [('ORDERS', oo), ('DAILY', [dict(date=str(t.date()), equity=float(v)) for t,v in eq.items()]), ('TRADES', tr), ('HOLDINGS', hold)]:
            dump(name+'_'+suffix+'.json', data)
        print(name, round(common['metrics']['cagr']*100, 3), round(common['metrics']['sharpe'], 3), flush=True)
    for name, oldname in [('original65', 'original65'), ('previous66', 'primary')]:
        old = read(ep.P/'RESULTS.json')[oldname]['metrics']
        for k in ['cagr', 'sharpe', 'max_drawdown']:
            assert abs(results[name]['metrics'][k]-old[k]) < 1e-10
    months = []
    for i, day in enumerate(m['dates']):
        row = dict(date=day, end=m['ends'][i], split='early53' if i < 53 else 'recent35' if i < 88 else 'later34')
        for name, oo in orders.items():
            if name == 'qqq': continue
            ss = oo[i]['symbols']
            ret = px.loc[m['ends'][i], ss]/px.loc[day, ss]-1 if ss else pd.Series(dtype=float)
            row[name] = dict(symbols=ss, monthly_return=float(ret.mean()) if ss else 0.)
        months.append(row)
    dump('MONTH_COMPARISON.json', months)
    attribution = {}
    for b in BASES:
        for name, ref in [(b+'_event', b), (b+'_ridge_earn', b),
            (b+'_event', b+'_matched_recent'), (b+'_ridge_earn', b+'_ridge_plain')]:
            changes = []
            for row in months:
                x, y = row[name], row[ref]
                if set(x['symbols']) != set(y['symbols']):
                    changes.append(dict(date=row['date'], split=row['split'], old=y['symbols'], new=x['symbols'],
                        difference=x['monthly_return']-y['monthly_return'],
                        log_contribution=float(np.log1p(x['monthly_return'])-np.log1p(y['monthly_return']))))
            positive = sorted([z['log_contribution'] for z in changes if z['log_contribution'] > 0], reverse=True)
            met = results[name]['metrics']
            undo = lambda n: float(((1+met['total_return'])*np.exp(-sum(positive[:n])))**(1/met['years'])-1)
            attribution[name+'_vs_'+ref] = dict(changed_months=len(changes), changes=changes,
                undo_best1_diagnostic_cagr=undo(1), undo_best3_diagnostic_cagr=undo(3))
    dump('ATTRIBUTION.json', attribution)
    changed = [f for f,h in read(P/'PROTECTED.json').items() if sha(f) != h]
    assert not changed, changed
    dump('RESULTS.json', results)
    dump('VALIDATION.json', dict(original_controls_exact=True, independent_fixed_unit_replay=True,
        frozen_train_result_parity=True, protected_files=len(read(P/'PROTECTED.json')), protected_changes=changed,
        original_eligibility_unchanged=True, author_labels_used=False, pristine_oos=False,
        observer_accounts_changed=False, scoped_event_data_admitted=True))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['initialize', 'prepare', 'forecasts', 'train', 'evaluate'])
    globals()[ap.parse_args().phase]()
