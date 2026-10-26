"""Cached factor composites preserving original support in the local boundary."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib.util
import json
import sys
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parent
O = P.parent
ROOT = O.parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


local = module('boundary_preserve70_local', O/'monthly_local_linear70/study.py')
scope, target = local.scope, local.target
old, rp, ex, arch, ep, read, sha, adapter = (local.old, local.rp, local.ex,
    local.arch, local.ep, local.read, local.sha, local.adapter)
placement = module('preserve70_placement', P/'placement.py')
TRAIN = 88
LOCAL_SPEC = dict(id=4, base='incumbent_scope67', mode='local_linear')
BOUNDARY = ROOT/'work/monitor_research_boundary/settlement_20261014T200412Z'
PROTECTED_COUNT = 21444
DESIGN_SHA256 = 'ae0a1a4c159c4da6bf4c078eb365fc8d0eb37268777c998d62c9571796e131ed'
ACTUAL_BOUND_FILES = ('DATA.npz', 'META.json', 'FEATURE_AUDIT.json',
                      'INPUT_HASHES.json', 'CACHE_HASHES.json', 'PLAN.json')
POLICY_FIELDS = dict(local.POLICY_FIELDS)
FACTOR_SOURCES = {
    'ordinal': (O/'monthly_ordinal_complexity70', 'rank_ordinal', 'ordinal_available'),
    'cotail': (O/'monthly_tail_cooccurrence70', 'rank_cotail', 'cotail_available'),
    'leadlag': (O/'monthly_sector_leadlag70', 'rank_leadlag', 'leadlag_available'),
}
FACTOR_FILES = ('DESIGN.json','DESIGN.sha256','PLAN.json','DATA.npz','META.json',
    'INCUMBENT_POLICY.json','FEATURE_AUDIT.json','INPUT_HASHES.json','CACHE_HASHES.json',
    'ACTUAL_FEATURE_REVIEW.json')
FACTOR_KEYS = tuple(key for _,rank,available in FACTOR_SOURCES.values()
                    for key in (rank,available))


def dump(name, value):
    (P/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else x)+'\n')


def configs():
    return [dict(id=0, base='original65', mode='none'),
            dict(id=1, base='incumbent66', mode='none'),
            dict(id=2, base='incumbent_scope67', mode='none'),
            dict(id=3, base='incumbent_local67', mode='none'),
            dict(id=4, base='incumbent_local67', mode='ordinal25_preserve'),
            dict(id=5, base='incumbent_local67', mode='cotail25_preserve'),
            dict(id=6, base='incumbent_local67', mode='leadlag25_preserve')]


def diagnostic_name(spec):
    if spec not in configs()[4:]:
        raise ValueError('Only preregistered support-preserving boundary candidates allowed')
    return spec['mode']


def implementations():
    return tuple(sorted(set(local.implementations()) |
        {P/name for name in ('study.py','test_study.py','report.py','placement.py','test_placement.py')} |
        {Path(adapter.__file__), placement.SOURCE_PLACEMENT_PATH,
         placement.SOURCE_PLACEMENT_PATH.with_name('test_placement.py'),
         Path(placement.path_adapter.__file__)}))


def source_files():
    files = set(local.source_files())
    files.update(local.P/name for name in ('DESIGN.json', 'DESIGN.sha256', 'PLAN.json',
        'GRID.json', 'FROZEN.json', 'FROZEN.sha256', 'DATA.npz', 'META.json', 'INPUT_HASHES.json',
        'CACHE_HASHES.json', 'INCUMBENT_POLICY.json', 'VALIDATION.json', 'RESULTS.json',
        'original65_ORDERS.json', 'incumbent66_ORDERS.json', 'incumbent_scope67_ORDERS.json',
        'local_linear_ORDERS.json'))
    files.update(BOUNDARY/name for name in ('BEFORE.json', 'AFTER.json',
        'RESEARCH_BOUNDARY.json', 'NEXT_EXPECTED_PROTECTED.json'))
    files.update((P/'placement.py',P/'test_placement.py',placement.SOURCE_PLACEMENT_PATH,
        placement.SOURCE_PLACEMENT_PATH.with_name('test_placement.py'),
        Path(placement.path_adapter.__file__)))
    files.update(folder/name for folder,_,_ in FACTOR_SOURCES.values() for name in FACTOR_FILES)
    return tuple(sorted(files))


def design():
    digest = (P/'DESIGN.sha256').read_text().strip()
    assert sha(P/'DESIGN.json') == digest == DESIGN_SHA256
    value = read(P/'DESIGN.json')
    assert value['configs'] == configs()
    assert value['factor_sources'] == {k:[folder.name,rank,available]
        for k,(folder,rank,available) in FACTOR_SOURCES.items()}
    assert value['boundary'] == str(BOUNDARY) and value['protected_count'] == PROTECTED_COUNT
    return value, digest


def verify_source():
    local.registered()
    local.verify_source()
    assert local.configs()[4] == LOCAL_SPEC
    assert sha(local.P/'FROZEN.json') == (local.P/'FROZEN.sha256').read_text().strip()
    old.assert_hashes(read(local.P/'FROZEN.json')['hashes'], local.P)
    old.assert_hashes(read(local.P/'INPUT_HASHES.json'))
    assert read(local.P/'FROZEN.json')['selected']['primary']['spec'] == LOCAL_SPEC
    old.assert_hashes(read(local.P/'CACHE_HASHES.json'), local.P)
    audit = read(local.P/'VALIDATION.json')
    assert audit['source_model_parity'] and audit['original_controls_exact']
    assert audit['independent_fixed_unit_replay'] and audit['protected_changes'] == []


def initialize():
    assert not (P/'PLAN.json').exists(), 'Never overwrite a registration'
    fixed, digest = design()
    verify_source()
    protected = read(BOUNDARY/'NEXT_EXPECTED_PROTECTED.json')
    assert len(protected) == PROTECTED_COUNT
    old.assert_hashes(protected)
    raw = read(local.P/'PLAN.json')['raw_source_hashes']
    old.assert_hashes(raw)
    for folder,_,_ in FACTOR_SOURCES.values():
        verify_factor_source(folder,raw)
    dump('PLAN.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        design_sha256=digest, design=fixed, raw_source_hashes=raw,
        calendar=local.calendar_provenance(), boundary=str(BOUNDARY), protected_count=PROTECTED_COUNT,
        hashes={str(f): sha(f) for f in implementations()},
        source_hashes={str(f): sha(f) for f in source_files()}))
    dump('GRID.json', configs())
    dump('PROTECTED.json', protected)


def registered():
    _, digest = design()
    assert read(P/'GRID.json') == configs()
    plan = read(P/'PLAN.json')
    assert plan['design_sha256'] == digest
    for key in ('hashes', 'source_hashes', 'raw_source_hashes'):
        old.assert_hashes(plan[key])
    assert plan['calendar'] == local.calendar_provenance()
    assert plan['boundary'] == str(BOUNDARY) and plan['protected_count'] == PROTECTED_COUNT


def assert_protected():
    old.assert_hashes(read(P/'PROTECTED.json'))


def source_spec(spec):
    if spec not in configs():
        raise ValueError('Only seven fixed configurations permitted')
    return local.configs()[spec['id']] if spec['id'] < 3 else LOCAL_SPEC


def factor_pair(rank, available, shape, n, label):
    """Validate only the requested prefix; no ranking or missing-value filling."""
    value=adapter._matrix(rank,shape,n,label+' rank')
    valid=adapter._matrix(available,shape,n,label+' availability',boolean=True)
    if np.isinf(value).any() or not np.array_equal(np.isfinite(value),valid):
        raise ValueError('Finite rank mask must equal availability: '+label)
    if (np.abs(value[valid])>.5).any():
        raise ValueError('Factor ranks must lie in [-.5,.5]: '+label)
    return value,valid


def feature_rank(spec, a, n):
    """Copy the selected cached rank; never inspect other arms or future values."""
    diagnostic_name(spec)
    shape=local._shape(a,n)
    label=spec['mode'].split('25_')[0]
    _,rank,available=FACTOR_SOURCES[label]
    value,_=factor_pair(a[rank],a[available],shape,n,label)
    result=np.full(shape,np.nan)
    result[:n]=value
    return result


def choices(spec, a, m, n, *, policy=None, boundary_selector=None):
    source = source_spec(spec)
    if spec['mode'] == 'none':
        return local.choices(source, a, m, n, policy=policy, boundary_selector=boundary_selector)
    policy = read(P/'INCUMBENT_POLICY.json') if policy is None else policy
    assert all(policy.get(k) == v for k, v in POLICY_FIELDS.items())
    pre = local.modified_pre_score(LOCAL_SPEC, a, n)
    return placement.score_and_choose(pre, feature_rank(spec, a, n), a, m, n,
        policy=policy,
        boundary_selector=ep.select_boundary if boundary_selector is None else boundary_selector)


def verify_inherited_arrays(current, source):
    """All27 original arrays retain dtype, shape and byte payload, including NaNs."""
    assert len(source) == 27, 'Unexpected inherited source array set'
    for key, expected in source.items():
        actual = current[key]
        assert actual.shape == expected.shape and actual.dtype == expected.dtype, key
        assert actual.tobytes() == expected.tobytes(), 'Inherited array drift: '+key


def verify_factor_source(folder, expected_raw):
    """Read prepared-source manifests/audit only, never its study or outcomes."""
    plan=read(folder/'PLAN.json')
    assert sha(folder/'DESIGN.json')==(folder/'DESIGN.sha256').read_text().strip()==plan['design_sha256']
    for key in ('hashes','source_hashes','raw_source_hashes'):
        old.assert_hashes(plan[key])
    assert len(expected_raw)==64 and plan['raw_source_hashes']==expected_raw
    old.assert_hashes(read(folder/'INPUT_HASHES.json'))
    old.assert_hashes(read(folder/'CACHE_HASHES.json'),folder)
    audit=read(folder/'ACTUAL_FEATURE_REVIEW.json')
    assert audit.get('passed') is True, 'Source actual feature audit has not passed'
    assert isinstance(audit.get('hashes'),dict), 'Source actual audit missing bindings'
    for name in ACTUAL_BOUND_FILES:
        assert audit['hashes'].get(name)==sha(folder/name), 'Source actual audit drift: '+name
    return dict(directory=str(folder),hashes={name:sha(folder/name) for name in ACTUAL_BOUND_FILES},
        actual_feature_review_sha256=sha(folder/'ACTUAL_FEATURE_REVIEW.json'),
        raw_source_hashes=plan['raw_source_hashes'])


def matched_features(caches, metas, source, meta):
    """Extract exact cached ranks/masks after validating every inherited byte."""
    assert set(caches)==set(metas)==set(FACTOR_SOURCES)
    shape=source['baseline'].shape
    selected={}
    for label,(_,rank,available) in FACTOR_SOURCES.items():
        assert metas[label]==meta, 'Source META drift: '+label
        verify_inherited_arrays(caches[label],source)
        value=np.asarray(caches[label][rank])
        if value.dtype.kind!='f':
            raise ValueError('Cached rank must retain a floating dtype: '+label)
        factor_pair(value,caches[label][available],shape,shape[0],label)
        selected[rank]=value.copy()
        selected[available]=np.asarray(caches[label][available]).copy()
    return selected


def prepare():
    registered()
    assert not (P/'DATA.npz').exists() and not (P/'FROZEN.json').exists()
    m,a,px,op=local.load()
    assert len(a)==27 and a['baseline'].shape==(122,49)
    inherited={key:value.copy() for key,value in a.items()}
    policy=read(local.P/'INCUMBENT_POLICY.json')
    expected_raw=read(P/'PLAN.json')['raw_source_hashes']
    bindings={};caches={};metas={}
    for label,(folder,rank,available) in FACTOR_SOURCES.items():
        bindings[label]=verify_factor_source(folder,expected_raw)
        assert read(folder/'INCUMBENT_POLICY.json')==policy, 'Source policy drift: '+label
        metas[label]=read(folder/'META.json')
        with np.load(folder/'DATA.npz',allow_pickle=False) as z:
            # Only declared ranks/masks and inherited arrays; no raw factors/outcomes.
            caches[label]={name:z[name].copy() for name in list(inherited)+[rank,available]}
    factors=matched_features(caches,metas,inherited,m)
    for key,value in factors.items():
        assert key not in a
        a[key]=value
    shape=a['baseline'].shape
    controls={}
    for spec in configs()[:4]:
        got=choices(spec,a,m,len(m['dates']),policy=policy)
        want=local.choices(source_spec(spec),a,m,len(m['dates']),policy=policy)
        assert got[0]==want[0]
        assert got[1].tobytes()==want[1].tobytes()
        filename=('original65','incumbent66','incumbent_scope67','local_linear')[spec['id']]
        assert rp.make_orders(*got,m)==read(local.P/(filename+'_ORDERS.json'))
        controls[spec['id']]=got
    missing=dict(a,**{rank:np.full(shape,np.nan) for _,rank,_ in FACTOR_SOURCES.values()},
        **{available:np.zeros(shape,bool) for _,_,available in FACTOR_SOURCES.values()})
    zeros=dict(a,**{rank:np.where(a[available],0.,np.nan)
                   for _,rank,available in FACTOR_SOURCES.values()})
    for spec in configs()[4:]:
        fallback=choices(spec,missing,m,len(m['dates']),policy=policy)
        assert fallback[0]==controls[3][0] and fallback[1].tobytes()==controls[3][1].tobytes()
        zero=choices(spec,zeros,m,len(m['dates']),policy=policy)
        assert zero[0]==controls[3][0] and zero[1].tobytes()==controls[3][1].tobytes()
        full=choices(spec,a,m,len(m['dates']),policy=policy)
        for n in (0,1,36,60,76,TRAIN,len(m['dates'])):
            prefix=choices(spec,a,m,n,policy=policy)
            assert prefix[0]==full[0][:n]
            np.testing.assert_array_equal(prefix[1],full[1][:n])
    verify_inherited_arrays(a,inherited)
    assert set(a)==set(inherited)|set(FACTOR_KEYS) and len(a)==33
    audit=dict(source_bindings=bindings,
        rank_counts={rank:int(np.isfinite(factors[rank]).sum())
                     for _,rank,_ in FACTOR_SOURCES.values()},
        mask_counts={available:int(factors[available].sum())
                     for _,_,available in FACTOR_SOURCES.values()},
        monthly_counts=[dict(date=day,
            rank_counts={rank:int(np.isfinite(factors[rank][i]).sum())
                         for _,rank,_ in FACTOR_SOURCES.values()},
            mask_counts={available:int(factors[available][i].sum())
                         for _,_,available in FACTOR_SOURCES.values()})
                        for i,day in enumerate(m['dates'])],
        hand='Exact source rank; .025*(clip(original_support+rank,-.5,.5)-original_support) in locked boundary only; zero/missing bypass',
        no_cross_sectional_ranking=True,no_feature_recomputation=True,
        independent_source_availability=True,source_ranks_and_masks_byte_exact=True,
        inherited_array_count=27,total_array_count=33,inherited_arrays_byte_exact=True,metas_exact=True,
        raw_source_count=64,original_local_pre_top2_and_cohort_locked=True)
    np.savez_compressed(P/'DATA.npz',**a)
    dump('META.json',m);dump('INCUMBENT_POLICY.json',policy);dump('FEATURE_AUDIT.json',audit)
    dump('PREPARE_VALIDATION.json',dict(four_controls_exact=True,prefix_exact=True,
        missing_masks_return_local_control=True,zero_ranks_return_local_control=True,
        original_core_and_auxiliary_states_preserved=True,
        source_raw_inputs=64,inherited_array_count=27,inherited_arrays_byte_exact=True,
        source_audit_bindings_verified=True,source_metas_exact=True,
        source_ranks_and_masks_exact=True,prepared_array_count=33,
        no_refit=True,no_reranking=True,no_price_download_or_reconstruction=True,
        execution_prices_from_exact_local_cache=True,placements=['boundary']))
    hashes=dict(read(local.P/'INPUT_HASHES.json'))
    for folder,_,_ in FACTOR_SOURCES.values():
        for key,value in read(folder/'INPUT_HASHES.json').items():
            assert key not in hashes or hashes[key]==value, 'Conflicting source dependency: '+key
            hashes[key]=value
    for f in list(implementations())+list(source_files())+[P/'DESIGN.json',P/'DESIGN.sha256']:
        value=sha(f)
        assert str(f) not in hashes or hashes[str(f)]==value
        hashes[str(f)]=value
    dump('INPUT_HASHES.json',hashes)
    dump('CACHE_HASHES.json',{f:sha(P/f) for f in ('DATA.npz','META.json','INCUMBENT_POLICY.json',
        'FEATURE_AUDIT.json','PREPARE_VALIDATION.json','INPUT_HASHES.json')})
    print('Prepared three cached support-preserving boundary candidates; four controls exact',flush=True)


def load():
    registered();verify_source()
    old.assert_hashes(read(P/'INPUT_HASHES.json'))
    old.assert_hashes(read(P/'CACHE_HASHES.json'),P)
    m,source,px,op=local.load()
    assert m == read(P/'META.json')
    with np.load(P/'DATA.npz',allow_pickle=False) as z:
        a={k:z[k].copy() for k in z.files}
    verify_inherited_arrays(a,source)
    assert set(a)==set(source)|set(FACTOR_KEYS) and len(a)==33
    for label,(_,rank,available) in FACTOR_SOURCES.items():
        factor_pair(a[rank],a[available],source['baseline'].shape,len(m['dates']),label)
    return m,a,px,op


def admitted(z, incumbent):
    return (z['training']['cagr'] >= incumbent['training']['cagr']-1e-12
        and z['training']['sharpe'] >= incumbent['training']['sharpe']-.05
        and z['training']['max_drawdown'] >= incumbent['training']['max_drawdown']-.03
        and z['early']['cagr'] >= incumbent['early']['cagr']-.025
        and z['recent']['cagr'] >= incumbent['recent']['cagr']-.025)


def select(rows):
    assert [z['spec'] for z in rows] == configs()
    incumbent = rows[3]
    for z in rows:
        z['admitted'] = admitted(z,incumbent)
    qualified = [incumbent]+[z for z in rows[4:] if z['admitted']]
    best = max(z['training']['cagr'] for z in qualified)
    primary = min((z for z in qualified if best-z['training']['cagr'] <= 1e-12),
                  key=lambda z:z['spec']['id'])
    selected = dict(primary=primary,original65=rows[0],incumbent66=rows[1],
        incumbent_scope67=rows[2],incumbent_local67=incumbent)
    selected.update({diagnostic_name(z['spec']):z for z in rows[4:]})
    return selected


def verify_actual_review():
    """Training requires an independent pass bound to all six prepared inputs."""
    audit = read(P/'ACTUAL_FEATURE_REVIEW.json')
    assert audit.get('passed') is True, 'Independent actual feature audit has not passed'
    hashes = audit.get('hashes')
    assert isinstance(hashes, dict), 'Independent audit must bind input hashes'
    for name in ACTUAL_BOUND_FILES:
        assert hashes.get(name) == sha(P/name), 'Independent audit input drift: '+name
    return audit


def train():
    registered()
    verify_actual_review()
    assert not (P/'FROZEN.json').exists(), 'Never overwrite frozen selection'
    m,a,px,_ = load()
    blocks,_,bounds,valid = rp.paths(px,m,TRAIN)
    rows,returns = [],[]
    for spec in read(P/'GRID.json'):
        ch,_ = choices(spec,a,m,TRAIN)
        rr = rp.fast_returns(ch,blocks,valid)
        rows.append(dict(spec=spec,**old.session.measure(rr,m,bounds)))
        returns.append(rr)
    selected = select(rows)
    folds,stitched,control = [],[],[]
    for n,end in ((36,60),(60,76),(76,88)):
        prefix = [dict(spec=z['spec'],**old.session.measure(rr,m,bounds,n)) for z,rr in zip(rows,returns)]
        winner = select(prefix)['primary']
        stitched.extend(returns[winner['spec']['id']][bounds[n]:bounds[end]])
        control.extend(returns[3][bounds[n]:bounds[end]])
        folds.append(dict(train_end=m['ends'][n-1],test_end=m['ends'][end-1],spec=winner['spec']))
    dump('WALKFORWARD.json',dict(folds=folds,pristine_oos=False,
        deployment='Each fold selects a fixed rule using only its training prefix; no dependence on previous holdings. Holdings remain monthly fixed units.',
        metrics=rp.stats_rr(np.array(stitched),m['dates'][36],m['ends'][87]),
        incumbent=rp.stats_rr(np.array(control),m['dates'][36],m['ends'][87])))
    dump('TRAIN_TRIALS.json',rows)
    np.savez_compressed(P/'TRAIN_RETURNS.npz',returns=np.stack(returns),bounds=np.array(bounds))
    files=('DESIGN.json','DESIGN.sha256','PLAN.json','GRID.json','DATA.npz','META.json','INCUMBENT_POLICY.json',
           'FEATURE_AUDIT.json','PREPARE_VALIDATION.json','ACTUAL_FEATURE_REVIEW.json',
           'INPUT_HASHES.json','CACHE_HASHES.json','PROTECTED.json','study.py','test_study.py','report.py',
           'TESTS.txt','WALKFORWARD.json','TRAIN_TRIALS.json','TRAIN_RETURNS.npz')
    frozen_hashes={f:sha(P/f) for f in files}
    # Bind prepared feature sources and their audits, not subsequent outcomes.
    for path in set(implementations()) | set(source_files()):
        frozen_hashes[str(path)]=sha(path)
    dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),selected=selected,
        trials=7,new_admitted=sum(int(z['admitted']) for z in rows[4:]),
        primary_source=selected['primary']['spec'],
        hashes=frozen_hashes,pristine_oos=False,author_labels_used=False,
        diagnostics_may_be_unqualified=True,all_diagnostics_frozen=True,cagr_tie_tolerance=1e-12))
    (P/'FROZEN.sha256').write_text(sha(P/'FROZEN.json')+'\n')
    print('Frozen primary',selected['primary']['spec'],flush=True)


def evaluate():
    registered()
    frozen=read(P/'FROZEN.json')
    assert sha(P/'FROZEN.json') == (P/'FROZEN.sha256').read_text().strip()
    old.assert_hashes(frozen['hashes'],P)
    m,a,px,op=load()
    blocks,_,bounds,valid=rp.paths(px,m,len(m['dates']))
    orders={}
    for name,z in frozen['selected'].items():
        ch,score=choices(z['spec'],a,m,len(m['dates']))
        orders[name]=rp.make_orders(ch,score,m)
    orders['qqq']=[dict(date=m['dates'][0],signal_date=m['dates'][0],end=m['ends'][-1],symbols=['QQQ'],scores=[])]
    results,memo={},{}
    for name,oo in orders.items():
        key=json.dumps(oo,sort_keys=True)
        if key not in memo:
            met,eq,tr,hold=ex.e.engine.replay(px,op,oo,cost=0.,execution='close',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
            for year in met['annual_returns']:
                if year['year'] in (2016,2026):year['partial']=True
            common=dict(metrics=met,slices=arch.old.prior.old.previous.cuts(eq,m))
            for label,cost in (('next_open_zero',0.),('next_open_cost',.0005)):
                mm,ee,_,_=ex.e.engine.replay(px,op,oo,cost=cost,execution='open',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
                for year in mm['annual_returns']:
                    if year['year'] in (2016,2026):year['partial']=True
                common[label]=dict(metrics=mm,slices=arch.old.prior.old.previous.cuts(ee,m))
            memo[key]=common,eq,tr,hold
        common,eq,tr,hold=memo[key]
        if name!='qqq':
            ch=[[m['symbols'][i].index(symbol) for symbol in row['symbols']] for i,row in enumerate(oo)]
            rr=rp.fast_returns(ch,blocks,valid)
            np.testing.assert_allclose(eq,np.r_[1.,np.cumprod(1+rr)]*100000,rtol=1e-11,atol=1e-5)
            assert abs(common['slices']['training2016_2023']['cagr']-frozen['selected'][name]['training']['cagr'])<1e-10
        results[name]=dict(common,selection=frozen['selected'].get(name))
        for suffix,value in (('ORDERS',oo),('DAILY',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()]),('TRADES',tr),('HOLDINGS',hold)):
            dump(name+'_'+suffix+'.json',value)
        print(name,round(common['metrics']['cagr']*100,3),round(common['metrics']['sharpe'],3),flush=True)
    for name,filename in (('original65','original65_ORDERS.json'),
            ('incumbent66','incumbent66_ORDERS.json'),('incumbent_scope67','primary_ORDERS.json')):
        assert orders[name]==read(scope.P/filename), 'Source control order drift: '+name
    assert orders['incumbent_local67']==read(local.P/'local_linear_ORDERS.json'),'Local control order drift'
    months=[]
    for i,day in enumerate(m['dates']):
        row=dict(date=day,end=m['ends'][i],split='early53' if i<53 else 'recent35' if i<88 else 'later34')
        for name,oo in orders.items():
            if name=='qqq':continue
            symbols=oo[i]['symbols']
            ret=px.loc[m['ends'][i],symbols]/px.loc[day,symbols]-1 if symbols else []
            row[name]=dict(symbols=symbols,monthly_return=float(np.mean(ret)) if len(ret) else 0.)
        months.append(row)
    dump('MONTH_COMPARISON.json',months)
    attribution={}
    for name,reference in [(diagnostic_name(z),reference) for z in configs()[4:] for reference in ('incumbent_local67','incumbent_scope67')]:
        changes=[]
        for row in months:
            z,b=row[name],row[reference]
            if set(z['symbols'])!=set(b['symbols']):
                changes.append(dict(date=row['date'],split=row['split'],old=b['symbols'],new=z['symbols'],
                    difference=z['monthly_return']-b['monthly_return'],
                    log_contribution=float(np.log1p(z['monthly_return'])-np.log1p(b['monthly_return']))))
        positive=sorted([z['log_contribution'] for z in changes if z['log_contribution']>0],reverse=True)
        met=results[name]['metrics']
        undo=lambda n:float(((1+met['total_return'])*np.exp(-sum(positive[:n])))**(1/met['years'])-1)
        attribution[name+'__vs__'+reference]=dict(changed_months=len(changes),changes=changes,
            undo_best1_diagnostic_cagr=undo(1),undo_best3_diagnostic_cagr=undo(3))
    dump('ATTRIBUTION.json',attribution)
    assert_protected()
    dump('RESULTS.json',results)
    dump('VALIDATION.json',dict(original_controls_exact=True,independent_fixed_unit_replay=True,
        training_freeze_parity=True,protected_files=len(read(P/'PROTECTED.json')),protected_changes=[],
        observer_accounts_changed=False,pristine_oos=False,fixed_configs=7,
        core_coefficients_unchanged=True,auxiliary_states_unchanged=True,
        original_forecasts_unchanged=True,no_refit=True,hand_budgets=[.025],
        source_masks_from_skip42_253price_windows=True,
        nominal_episode_budget=.05,all_diagnostics_frozen=True,
        boundary_original_local_anchors_fixed=True,boundary_original_local_candidate_cohort_fixed=True,
        hand_budget_applied_once=True,original_support_payload_exact=True,
        true_episode_boundary_rules=True,no_global_rerank=True,
        no_intramonth_transactions=True,annual_clock_unchanged=True,
        raw_source_and_execution_cache_hashes_verified=True,support_preserving_composites=True,
        cached_ranks_not_reranked=True,zero_and_missing_exact_identity=True,source_audits_bound=True,
        source_ranks_and_masks_exact=True,inherited_array_count=27,prepared_array_count=33,
        independent_actual_feature_audit_required_before_train=True,
        primary_target70_reached=results['primary']['metrics']['cagr']>=.70))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('initialize','prepare','train','evaluate'))
    globals()[parser.parse_args().phase]()
