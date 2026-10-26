"""Two preregistered monthly Ridge uncertainty-shrink experiments; research only."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
DESIGN_SHA = 'e2a5318e2ad34ba034e98416e8fe55d89c564bac1816de988c0396b24e3bb569'
NAMES = ('incumbent_local67', 'cluster_sensitivity_shrink')
TRAIN = 88


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


local = module('cluster70_local', P.parent/'monthly_local_linear70/study.py')
rp, ex, old, arch, target = local.rp, local.ex, local.old, local.arch, local.target
LOCAL_SPEC = local.configs()[4]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else x)+'\n'
    dest = P/name
    if dest.exists():
        if dest.read_text() != text:
            raise ValueError('Refusing to overwrite different artifact: '+name)
    else:
        with dest.open('x') as out:
            out.write(text)


def registered():
    assert sha(P/'DESIGN.json') == DESIGN_SHA
    plan = read(P/'PLAN.json')
    assert plan['design_sha256'] == DESIGN_SHA
    old.assert_hashes(plan['input_hashes'])
    return plan


def initialize():
    assert not (P/'PLAN.json').exists()
    assert sha(P/'DESIGN.json') == DESIGN_SHA
    design = read(P/'DESIGN.json')
    for path, digest in design['source_hashes'].items():
        assert sha(ROOT/path) == digest
    local.registered()
    local.verify_source()
    sources = {str(f): sha(f) for f in local.implementations()}
    sources.update({str(P/name): sha(P/name) for name in ('study.py','test_study.py','model.py','test_model.py')})
    sources.update({str(ROOT/path): digest for path,digest in design['source_hashes'].items()})
    sources.update({str(local.P/name): sha(local.P/name) for name in ('INPUT_HASHES.json','CACHE_HASHES.json')})
    sources[str(P/'TESTS.txt')] = sha(P/'TESTS.txt')
    sources[str(P/'DESIGN.json')] = DESIGN_SHA
    boundary = design['protected_manifest']
    assert sha(ROOT/boundary['path']) == boundary['sha256']
    old.assert_hashes(read(ROOT/boundary['path']))
    write('PLAN.json', dict(registered_utc=datetime.now(timezone.utc).isoformat(),
          design_sha256=DESIGN_SHA, input_hashes=sources,
          protected_manifest=boundary, configs=design['configs'], actual_fits_before_registration=0))
    print('Registered two fixed arms', flush=True)


def ranks(raw, source_raw, a):
    shape = a['baseline'].shape
    raw = np.asarray(raw).reshape(shape)
    source_raw = np.asarray(source_raw).reshape(shape)
    use = a['forecast_available'] & a['eligible']
    assert np.array_equal(np.isfinite(raw), a['forecast_available'])
    result = a['rank_log_excess'].copy()
    for i in range(len(use)):
        if not use[i].any() or np.array_equal(raw[i,use[i]], source_raw[i,use[i]]):
            continue
        r = local.ep.old.prior.ranks(pd.Series(raw[i]).where(use[i]))
        result[i,use[i]] = r[use[i]]
    return result


def pre_score(name, a, n):
    if name not in NAMES:
        raise ValueError('Unregistered arm')
    result = local.modified_pre_score(LOCAL_SPEC, a, n)
    if name == NAMES[0]:
        return result
    use = a['forecast_available'][:n] & a['eligible'][:n]
    weak, calm = a['state_weak_narrow'][:n], a['state_calm10'][:n]
    weight = .25*(.05*weak[:,None] + .01*(calm & ~weak)[:,None]*a['extension_available'][:n])
    delta = weight*(a['rank_'+name][:n]-a['rank_log_excess'][:n])
    changed = use & (delta != 0)
    result[:n][changed] += delta[changed]
    return result


def choices(name, a, m, n):
    if name == NAMES[0]:
        return local.choices(LOCAL_SPEC, a, m, n)
    return local.adapter.score_and_choose('incumbent66', 0., np.full(a['baseline'].shape,np.nan),
        a,m,n,pre_score=pre_score(name,a,n),effects=a['effects'],
        episode_policy=read(local.P/'INCUMBENT_POLICY.json'), boundary_selector=local.ep.select_boundary)


def prepare():
    registered()
    assert not (P/'DATA.npz').exists()
    m,a,_,_ = local.load()
    model = module('cluster70_model', P/'model.py')
    with np.load(target.P/'TRAIN_INPUTS.npz', allow_pickle=False) as z:
        x = {k:z[k].copy() for k in z.files}
    forecast = model.annual_forecasts(x['dates'],x['ends'],x['train_plain'],
        {k:x['target_'+k] for k in target.TARGETS},x['common_good'],
        x['event_dates'],x['event_plain'],a['eligible'].reshape(-1)&np.isfinite(x['event_plain']).all(axis=1))
    shape = a['baseline'].shape
    np.testing.assert_array_equal(forecast['available'].reshape(shape),a['forecast_available'])
    source = forecast['raw']['source_log'].reshape(shape)
    np.testing.assert_allclose(source,a['raw_log_excess'],rtol=0,atol=1e-12,equal_nan=True)
    np.testing.assert_array_equal(local.source_ranked_forecast(source,a),a['rank_log_excess'])
    for name in NAMES[1:]:
        a['raw_'+name] = forecast['raw'][name].reshape(shape)
        a['rank_'+name] = ranks(a['raw_'+name],source,a)
    orders = {}
    for name in NAMES:
        ch,score = choices(name,a,m,len(m['dates']))
        short,sc = choices(name,a,m,TRAIN)
        assert short == ch[:TRAIN]
        np.testing.assert_array_equal(sc,score[:TRAIN])
        orders[name] = rp.make_orders(ch,score,m)
    assert orders[NAMES[0]] == read(local.P/'local_linear_ORDERS.json')
    np.savez_compressed(P/'DATA.npz',**a)
    write('META.json',m)
    write('MODEL_AUDIT.json',forecast['audit'])
    write('PRETRAIN_ORDERS.json',orders)
    write('PREPARE_VALIDATION.json',dict(source_raw_parity_atol=1e-12,source_rank_exact=True,
         source_availability_exact=True,control_orders_exact=True,prefix_choices_scores_exact=True,
         original_arrays_preserved=True,model_slot_fraction=.25,no_new_budget=True))
    write('CACHE_HASHES.json',{name:sha(P/name) for name in ('DATA.npz','META.json','MODEL_AUDIT.json',
          'PRETRAIN_ORDERS.json','PREPARE_VALIDATION.json')})
    registered()
    print('Prepared two arms and froze all orders before portfolio outcomes',flush=True)


def load():
    registered()
    old.assert_hashes(read(P/'CACHE_HASHES.json'),P)
    m,original,px,op = local.load()
    assert m == read(P/'META.json')
    with np.load(P/'DATA.npz',allow_pickle=False) as z:
        a = {k:z[k].copy() for k in z.files}
    for k in original:
        assert a[k].dtype == original[k].dtype and a[k].shape == original[k].shape
        assert a[k].tobytes() == original[k].tobytes()
    return m,a,px,op


def select(rows):
    incumbent = rows[0]
    accepted = [z for z in rows if local.admitted(z,incumbent)]
    best = max(z['training']['cagr'] for z in accepted)
    return min((z for z in accepted if best-z['training']['cagr']<=1e-12),key=lambda z:z['id'])


def train():
    assert not (P/'FROZEN.json').exists()
    m,a,px,_ = load()
    blocks,_,bounds,valid = rp.paths(px,m,TRAIN)
    rows,returns = [],[]
    for ident,name in enumerate(NAMES):
        ch,_ = choices(name,a,m,TRAIN)
        rr = rp.fast_returns(ch,blocks,valid)
        rows.append(dict(id=ident,name=name,**old.session.measure(rr,m,bounds)))
        returns.append(rr)
    winner = select(rows)
    folds,stitched,reference = [],[],[]
    for n,end in ((36,60),(60,76),(76,88)):
        prefix = [dict(id=i,name=NAMES[i],**old.session.measure(rr,m,bounds,n)) for i,rr in enumerate(returns)]
        pick = select(prefix)
        stitched.extend(returns[pick['id']][bounds[n]:bounds[end]])
        reference.extend(returns[0][bounds[n]:bounds[end]])
        folds.append(dict(train_months=n,test_end_month=end,selected=pick['name']))
    write('TRAIN_TRIALS.json',rows)
    write('WALKFORWARD.json',dict(folds=folds,metrics=rp.stats_rr(np.array(stitched),m['dates'][36],m['ends'][87]),
        incumbent=rp.stats_rr(np.array(reference),m['dates'][36],m['ends'][87]),pristine_oos=False))
    write('FROZEN.json',dict(selected=winner,configs=list(NAMES),trials=2,pristine_oos=False,
        cache_sha256=sha(P/'CACHE_HASHES.json'),orders_sha256=sha(P/'PRETRAIN_ORDERS.json'),
        plan_sha256=sha(P/'PLAN.json'),train_sha256=sha(P/'TRAIN_TRIALS.json'),
        frozen_utc=datetime.now(timezone.utc).isoformat()))
    print('Frozen first88 winner: '+winner['name'],flush=True)


def evaluate():
    m,a,px,op = load()
    frozen = read(P/'FROZEN.json')
    for key,name in [('cache_sha256','CACHE_HASHES.json'),('orders_sha256','PRETRAIN_ORDERS.json'),
                     ('plan_sha256','PLAN.json'),('train_sha256','TRAIN_TRIALS.json')]:
        assert frozen[key] == sha(P/name)
    orders = read(P/'PRETRAIN_ORDERS.json')
    blocks,_,bounds,valid = rp.paths(px,m,len(m['dates']))
    rows = read(P/'TRAIN_TRIALS.json')
    results = {}
    for i,name in enumerate(NAMES):
        ch,score = choices(name,a,m,len(m['dates']))
        oo = rp.make_orders(ch,score,m)
        assert oo == orders[name]
        met,eq,tr,hold = ex.e.engine.replay(px,op,oo,cost=0.,execution='close',
             start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
        rr = rp.fast_returns(ch,blocks,valid)
        np.testing.assert_allclose(eq,np.r_[1.,np.cumprod(1+rr)]*100000,rtol=1e-11,atol=1e-5)
        for yy in met['annual_returns']:
            if yy['year'] in (2016,2026):yy['partial']=True
        cuts = arch.old.prior.old.previous.cuts(eq,m)
        assert abs(cuts['training2016_2023']['cagr']-rows[i]['training']['cagr'])<1e-10
        item = dict(metrics=met,slices=cuts,admitted=local.admitted(rows[i],rows[0]))
        for label,cost in [('next_open_zero',0.),('next_open_cost',.0005)]:
            mm,ee,_,_ = ex.e.engine.replay(px,op,oo,cost=cost,execution='open',
                  start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
            for yy in mm['annual_returns']:
                if yy['year'] in (2016,2026):yy['partial']=True
            item[label] = dict(metrics=mm,slices=arch.old.prior.old.previous.cuts(ee,m))
        results[name] = item
        for suffix,value in [('ORDERS',oo),('TRADES',tr),('HOLDINGS',hold),
               ('DAILY',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()])]:
            write(name+'_'+suffix+'.json',value)
        print(name,met['cagr'],met['sharpe'],flush=True)
    assert abs(results[NAMES[0]]['metrics']['cagr']-read(local.P/'RESULTS.json')['primary']['metrics']['cagr'])<1e-12
    results['primary'] = dict(results[frozen['selected']['name']],selected_name=frozen['selected']['name'])
    write('RESULTS.json',results)
    changes = {}
    for name in NAMES[1:]:
        records = []
        for i,(base,new) in enumerate(zip(orders[NAMES[0]],orders[name])):
            if set(base['symbols']) == set(new['symbols']):continue
            def ret(row):
                ss = row['symbols']
                return float((px.loc[row['end'],ss]/px.loc[row['date'],ss]-1).mean()) if ss else 0.
            rb,rn = ret(base),ret(new)
            records.append(dict(date=base['date'],split='first88' if i<88 else 'later34',
                 old=base['symbols'],new=new['symbols'],old_return=rb,new_return=rn,
                 log_contribution=float(np.log1p(rn)-np.log1p(rb))))
        positive = sorted([r['log_contribution'] for r in records if r['log_contribution']>0],reverse=True)
        met = results[name]['metrics']
        changes[name] = dict(changed_months=len(records),changes=records,
            undo_best1_cagr=float(((1+met['total_return'])*np.exp(-sum(positive[:1])))**(1/met['years'])-1),
            undo_best3_cagr=float(((1+met['total_return'])*np.exp(-sum(positive[:3])))**(1/met['years'])-1))
    write('ATTRIBUTION.json',changes)
    boundary = read(P/'PLAN.json')['protected_manifest']
    old.assert_hashes(read(ROOT/boundary['path']))
    write('VALIDATION.json',dict(fixed_unit_replay=True,training_freeze_parity=True,
        control_exact=True,no_intramonth_change=True,observer_changed=False,
        protected_files=len(read(ROOT/boundary['path'])),target_reached=results['primary']['metrics']['cagr']>=.70))


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('phase',choices=('initialize','prepare','train','evaluate'))
    globals()[cli.parse_args().phase]()
