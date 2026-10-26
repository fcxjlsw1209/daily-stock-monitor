"""Fixed allocation sensitivity of already-frozen cluster Ridge ranks; no refits."""
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
DESIGN_SHA = '4070e4348ded409c66c0d99d5f6de19143bfaafd096892413e85852446a1cd4c'
NAMES = ('incumbent_local67', 'cluster_half_existing_slot', 'cluster_all_available_global_slot')
FRACTIONS = (0., .50, .75)
TRAIN = 88
EXPECTED_ARRAYS = 29
PROTECTED_FILES = 21444


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


source = module('cluster_budget70_source', P.parent/'monthly_cluster_shrink70/study.py')
local = source.local
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


def configs():
    return [dict(id=i, name=name, fraction=FRACTIONS[i]) for i, name in enumerate(NAMES)]


def design():
    assert sha(P/'DESIGN.json') == DESIGN_SHA, 'Registered design changed'
    fixed = read(P/'DESIGN.json')
    assert fixed['configs'] == configs()
    assert fixed['model']['refit'] is False
    return fixed


def protected(fixed, *, members=False):
    boundary = fixed['protected_manifest']
    path = ROOT/boundary['path']
    assert sha(path) == boundary['sha256'], 'Protected manifest itself changed'
    values = read(path)
    assert len(values) == PROTECTED_FILES, 'Protected membership count changed'
    if members:
        old.assert_hashes(values)
    return path


def registered():
    fixed = design()
    plan = read(P/'PLAN.json')
    assert plan['design_sha256'] == DESIGN_SHA
    assert plan['configs'] == configs()
    assert plan['actual_fits_before_registration'] == 0
    assert plan['protected_manifest'] == fixed['protected_manifest']
    manifest = protected(fixed)
    assert plan['input_hashes'].get(str(manifest)) == fixed['protected_manifest']['sha256']
    for path, digest in fixed['source_hashes'].items():
        assert plan['input_hashes'].get(str(ROOT/path)) == digest
    old.assert_hashes(plan['input_hashes'])
    return plan


def initialize():
    assert not (P/'PLAN.json').exists()
    fixed = design()
    for path, digest in fixed['source_hashes'].items():
        assert sha(ROOT/path) == digest
    source.registered()
    local.verify_source()
    sources = {str(f): sha(f) for f in local.implementations()}
    sources.update({str(source.P/name): sha(source.P/name) for name in ('study.py', 'test_study.py')})
    sources.update({str(P/name): sha(P/name) for name in
                    ('study.py', 'test_study.py', 'TESTS.txt', 'SOURCE_SLOT_REVIEW.json')})
    sources.update({str(ROOT/path): digest for path, digest in fixed['source_hashes'].items()})
    sources.update({str(local.P/name): sha(local.P/name) for name in ('INPUT_HASHES.json', 'CACHE_HASHES.json')})
    sources[str(P/'DESIGN.json')] = DESIGN_SHA
    manifest = protected(fixed, members=True)
    sources[str(manifest)] = fixed['protected_manifest']['sha256']
    write('PLAN.json', dict(registered_utc=datetime.now(timezone.utc).isoformat(),
          design_sha256=DESIGN_SHA, input_hashes=sources,
          protected_manifest=fixed['protected_manifest'], configs=configs(),
          actual_fits_before_registration=0, model_refit=False))
    print('Registered three fixed allocation arms; no model fits', flush=True)


def pre_score(name, a, n):
    if name not in NAMES:
        raise ValueError('Unregistered arm')
    shape = local._shape(a, n)
    result = local.modified_pre_score(LOCAL_SPEC, a, n)
    fraction = FRACTIONS[NAMES.index(name)]
    if fraction == 0.:
        return result
    _, use = local.original_rank_inputs(a, n)
    weak = local.adapter._state(a, 'state_weak_narrow', shape[0], n)
    calm = local.adapter._state(a, 'state_calm10', shape[0], n)
    extension = local.adapter._matrix(a['extension_available'], shape, n, 'extension availability', boolean=True)
    coefficient = fraction*(.05*weak[:, None] + .01*(calm & ~weak)[:, None]*extension)
    cluster = local.adapter._matrix(a['rank_cluster_sensitivity_shrink'], shape, n, 'frozen cluster rank')
    global_rank = local.adapter._matrix(a['rank_log_excess'], shape, n, 'frozen global rank')
    active = use & (coefficient != 0.)
    if not np.isfinite(cluster[active]).all() or (np.abs(cluster[active]) > .5+1e-12).any():
        raise ValueError('Active frozen cluster ranks must be finite and bounded')
    delta = np.zeros((n, shape[1]))
    delta[active] = coefficient[active]*(cluster[active]-global_rank[active])
    changed = delta != 0.
    view = result[:n].copy()
    view[changed] += delta[changed]
    result[:n] = view
    return result


def choices(name, a, m, n):
    if name not in NAMES:
        raise ValueError('Unregistered arm')
    if name == NAMES[0]:
        return local.choices(LOCAL_SPEC, a, m, n)
    return local.adapter.score_and_choose('incumbent66', 0., np.full(a['baseline'].shape, np.nan),
        a, m, n, pre_score=pre_score(name, a, n), effects=a['effects'],
        episode_policy=read(local.P/'INCUMBENT_POLICY.json'), boundary_selector=local.ep.select_boundary)


def assert_arrays_identical(cached, original):
    assert set(cached) == set(original) and len(cached) == EXPECTED_ARRAYS, 'Changed frozen array membership'
    for key in original:
        assert cached[key].dtype == original[key].dtype, 'Changed dtype: '+key
        assert cached[key].shape == original[key].shape, 'Changed shape: '+key
        assert cached[key].tobytes() == original[key].tobytes(), 'Changed frozen bytes: '+key


def source_data():
    fixed = design()
    for name in ('DATA.npz', 'META.json'):
        path = source.P/name
        relative = str(path.relative_to(ROOT))
        assert sha(path) == fixed['source_hashes'][relative], 'Frozen source changed: '+name
    m, a, px, op = source.load()
    with np.load(source.P/'DATA.npz', allow_pickle=False) as z:
        original = {key: z[key].copy() for key in z.files}
    assert_arrays_identical(a, original)
    assert m == read(source.P/'META.json')
    return m, a, px, op


def prepare():
    registered()
    assert not (P/'DATA.npz').exists()
    assert not (P/'META.json').exists()
    m, a, _, _ = source_data()
    orders = {}
    for name in NAMES:
        ch, score = choices(name, a, m, len(m['dates']))
        short, sc = choices(name, a, m, TRAIN)
        assert short == ch[:TRAIN]
        np.testing.assert_array_equal(sc, score[:TRAIN])
        orders[name] = rp.make_orders(ch, score, m)
    assert orders[NAMES[0]] == read(source.P/'incumbent_local67_ORDERS.json')
    assert orders[NAMES[0]] == read(local.P/'local_linear_ORDERS.json')
    for name in ('DATA.npz', 'META.json'):
        with (P/name).open('xb') as out:
            out.write((source.P/name).read_bytes())
        assert sha(P/name) == sha(source.P/name)
    write('SOURCE_REUSE.json', dict(actual_model_fits=0, source_arrays=EXPECTED_ARRAYS,
        source_data_sha256=sha(source.P/'DATA.npz'), source_meta_sha256=sha(source.P/'META.json'),
        source_model_audit_sha256=sha(source.P/'MODEL_AUDIT.json'), copied_byte_identically=True))
    write('PRETRAIN_ORDERS.json', orders)
    write('PREPARE_VALIDATION.json', dict(source_files_byte_identical=True, source_array_count=EXPECTED_ARRAYS,
        source_ranks_exact=True, source_availability_exact=True, control_orders_exact=True,
        prefix_choices_scores_exact=True, original_arrays_preserved=True, model_refit=False,
        actual_model_fits=0, model_slot_fractions=list(FRACTIONS), no_new_budget=True))
    write('CACHE_HASHES.json', {name: sha(P/name) for name in ('DATA.npz', 'META.json', 'SOURCE_REUSE.json',
          'PRETRAIN_ORDERS.json', 'PREPARE_VALIDATION.json')})
    registered()
    print('Copied 29 frozen arrays and froze all three arms before portfolio outcomes', flush=True)


def load():
    registered()
    old.assert_hashes(read(P/'CACHE_HASHES.json'), P)
    m, original, px, op = source_data()
    for name in ('DATA.npz', 'META.json'):
        assert sha(P/name) == sha(source.P/name), 'Copied frozen source changed: '+name
    assert m == read(P/'META.json')
    with np.load(P/'DATA.npz', allow_pickle=False) as z:
        a = {key: z[key].copy() for key in z.files}
    assert_arrays_identical(a, original)
    return m, a, px, op


def select(rows):
    assert [(z['id'], z['name']) for z in rows] == list(enumerate(NAMES)), 'Unregistered selection rows'
    incumbent = rows[0]
    accepted = [z for z in rows if local.admitted(z,incumbent)]
    best = max(z['training']['cagr'] for z in accepted)
    return min((z for z in accepted if best-z['training']['cagr']<=1e-12),key=lambda z:z['id'])


def train():
    assert not (P/'FROZEN.json').exists()
    m,a,px,_ = load()
    orders = read(P/'PRETRAIN_ORDERS.json')
    assert set(orders) == set(NAMES)
    blocks,_,bounds,valid = rp.paths(px,m,TRAIN)
    rows,returns = [],[]
    for ident,name in enumerate(NAMES):
        ch,score = choices(name,a,m,TRAIN)
        assert rp.make_orders(ch,score,m) == orders[name][:TRAIN]
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
    write('FROZEN.json',dict(selected=winner,configs=list(NAMES),trials=3,pristine_oos=False,
        cache_sha256=sha(P/'CACHE_HASHES.json'),orders_sha256=sha(P/'PRETRAIN_ORDERS.json'),
        plan_sha256=sha(P/'PLAN.json'),train_sha256=sha(P/'TRAIN_TRIALS.json'),
        frozen_utc=datetime.now(timezone.utc).isoformat()))
    print('Frozen first88 winner: '+winner['name'],flush=True)


def evaluate():
    m,a,px,op = load()
    protected(design(), members=True)
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
    protected(design(), members=True)
    write('VALIDATION.json',dict(fixed_unit_replay=True,training_freeze_parity=True,
        control_exact=True,no_intramonth_change=True,observer_changed=False,
        protected_files=PROTECTED_FILES,protected_manifest_self_hash=True,actual_model_fits=0,target_reached=results['primary']['metrics']['cagr']>=.70))


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('phase',choices=('initialize','prepare','train','evaluate'))
    globals()[cli.parse_args().phase]()
