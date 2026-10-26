"""Two fixed, zero-preserving auxiliary shifts using immutable feature caches.

No feature refit, remapping, common-mask restriction, or component-performance
selection. The same repeatedly studied history is not an independent holdout.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib.util
import json
import sys
import numpy as np
import pandas as pd

sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent
O=P.parent


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


upstream=module('aux_shift70_scope',O/'monthly_model_scope70/study.py')
ulcer=module('aux_shift70_ulcer',O/'monthly_ulcer70/study.py')
bond=module('aux_shift70_bond',O/'monthly_bond_response70/study.py')
old,rp,ex,arch,ep,read,sha,adapter=(upstream.old,upstream.rp,upstream.ex,
    upstream.arch,upstream.ep,upstream.read,upstream.sha,upstream.adapter)
TRAIN=88
POLICY_FIELDS=dict(upstream.POLICY_FIELDS)
SOURCE_PRIMARY_ID=3
SOURCE_PRIMARY_SPEC=dict(id=3,base='incumbent67',scope='calm_nonweak',budget=.01)
DESIGN_SHA256='f58e1d5dd1f9c5c382bd7c7a0662e3ff9e20d9627be2e349f5e83120fc299160'


def dump(name,value):
    (P/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else x)+'\n')


def configs():
    return [dict(id=0,base='original65',mode='none'),
            dict(id=1,base='incumbent66',mode='none'),
            dict(id=2,base='incumbent_scope67',mode='none'),
            dict(id=3,base='incumbent_scope67',mode='ulcer_shift'),
            dict(id=4,base='incumbent_scope67',mode='bond_shift')]


def diagnostic_name(spec):
    if spec not in configs()[3:]:raise ValueError('Only fixed candidates have diagnostic names')
    return spec['mode']


def components():return ulcer,bond


def implementations():
    files=set(upstream.implementations())|{P/'study.py',P/'test_study.py'}
    for study in components():files.update(study.implementations())
    return tuple(sorted(files))


def source_files():
    component_names=('DESIGN.json','DESIGN.sha256','PLAN.json','GRID.json','INPUT_HASHES.json',
        'CACHE_HASHES.json','FROZEN.json','FROZEN.sha256','DATA.npz','META.json',
        'INCUMBENT_POLICY.json','PROTECTED.json','FEATURE_AUDIT.json','PLACEMENT_AUDIT.json',
        'PREPARE_VALIDATION.json','VALIDATION.json','original65_ORDERS.json',
        'incumbent66_ORDERS.json','incumbent_scope67_ORDERS.json')
    files=[study.P/name for study in components() for name in component_names]
    files += [upstream.P/name for name in ('DESIGN.json','PLAN.json','GRID.json','INPUT_HASHES.json',
        'CACHE_HASHES.json','FROZEN.json','FROZEN.sha256','DATA.npz','META.json','INCUMBENT_POLICY.json',
        'PROTECTED.json','SCOPE_AUDIT.json','PREPARE_VALIDATION.json',
        'original65_ORDERS.json','incumbent66_ORDERS.json','primary_ORDERS.json')]
    for study in components():
        files += [Path(path) for path in study.features.calendar_provenance()['source_hashes']]
    return tuple(sorted(set(files)))


def design():
    assert sha(P/'DESIGN.json')==DESIGN_SHA256,'Frozen design changed'
    assert (P/'DESIGN.sha256').read_text().strip()==DESIGN_SHA256
    fixed=read(P/'DESIGN.json');assert fixed['grid']==configs()
    return fixed


def validate_component_completion(validation):
    for key in ('original_controls_exact','independent_fixed_unit_replay','training_freeze_parity'):
        assert validation.get(key) is True,'Component validation missing success: '+key
    assert validation.get('fixed_configs')==5
    assert validation.get('protected_files')==21436
    assert validation.get('protected_changes')==[]
    assert validation.get('observer_accounts_changed') is False


def assert_original_arrays(original,component):
    for key,value in original.items():
        assert key in component,'Missing inherited array: '+key
        left,right=np.asarray(value),np.asarray(component[key])
        assert left.dtype==right.dtype,'Inherited dtype changed: '+key
        np.testing.assert_array_equal(left,right,err_msg='Inherited source changed: '+key)


def validate_raw_hashes(raw,manifests):
    assert bond.raw_source_hashes(dict(source_hashes=raw))==raw
    for manifest in manifests:
        for path,digest in raw.items():
            assert manifest.get(path)==digest,'Raw input hash differs or absent: '+path
    return dict(raw)


def validate_cached_features(a,shape):
    for study,prefix in ((ulcer,'ulcer_'),(bond,'bond_')):
        arrays={key.removeprefix(prefix):value for key,value in a.items() if key.startswith(prefix)}
        study.validate_features(arrays,shape)


def source_bundles():
    """Validate shared axes, every inherited array, execution prices and provenance."""
    m,a,px,op=upstream.load()
    policy=read(upstream.P/'INCUMBENT_POLICY.json')
    protected=read(upstream.P/'PROTECTED.json');assert len(protected)==21436
    bundles=[];plans=[]
    for study in components():
        cm,ca,cp,co=study.load()
        assert cm==m,'Component metadata differs'
        assert read(study.P/'INCUMBENT_POLICY.json')==policy,'Component policy differs'
        assert read(study.P/'PROTECTED.json')==protected,'Component protection differs'
        pd.testing.assert_frame_equal(cp,px,check_exact=True)
        pd.testing.assert_frame_equal(co,op,check_exact=True)
        assert_original_arrays(a,ca)
        prefix='ulcer_' if study is ulcer else 'bond_'
        study.validate_features({k.removeprefix(prefix):v for k,v in ca.items() if k.startswith(prefix)},np.asarray(a['baseline']).shape)
        plan=read(study.P/'PLAN.json');audit=read(study.P/'FEATURE_AUDIT.json')
        assert study.features.calendar_provenance()==plan['calendar']==audit['calendar']
        plans.append(plan);bundles.append(ca)
    assert_original_arrays({k:bundles[0][k] for k in set(bundles[0])&set(bundles[1])},bundles[1])
    assert plans[0]['raw_source_hashes']==plans[1]['raw_source_hashes']
    assert plans[0]['calendar']==plans[1]['calendar']
    raw=plans[0]['raw_source_hashes']
    validate_raw_hashes(raw,[read(study.P/'INPUT_HASHES.json') for study in (upstream,*components())])
    return m,a,px,op,policy,bundles,raw,plans[0]['calendar']


def verify_source():
    upstream.registered()
    assert upstream.configs()[SOURCE_PRIMARY_ID]==SOURCE_PRIMARY_SPEC
    assert sha(upstream.P/'FROZEN.json')==(upstream.P/'FROZEN.sha256').read_text().strip()
    frozen=read(upstream.P/'FROZEN.json')
    assert frozen['selected']['primary']['spec']==SOURCE_PRIMARY_SPEC
    old.assert_hashes(frozen['hashes'],upstream.P)
    for study in components():
        study.registered();study.verify_source()
        assert sha(study.P/'FROZEN.json')==(study.P/'FROZEN.sha256').read_text().strip()
        # Read hashes only; the component winner and outcomes are never used.
        old.assert_hashes(read(study.P/'FROZEN.json')['hashes'],study.P)
        old.assert_hashes(read(study.P/'INPUT_HASHES.json'))
        old.assert_hashes(read(study.P/'CACHE_HASHES.json'),study.P)
        validate_component_completion(read(study.P/'VALIDATION.json'))
    return source_bundles()


def initialize():
    assert not (P/'PLAN.json').exists(),'Never overwrite registration'
    fixed=design();*_,raw,calendar=verify_source()
    protected=read(upstream.P/'PROTECTED.json')
    dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),design=fixed,
        design_created_at=fixed['created_at'],design_sha256=DESIGN_SHA256,
        hashes={str(f):sha(f) for f in implementations()},source_hashes={str(f):sha(f) for f in source_files()},
        raw_source_hashes=raw,calendar=calendar,
        source_arrays_metadata_prices_policy_raw_calendar_checked_before_registration=True,
        component_performance_used_as_algorithm_input=False,component_winner_used=False,no_feature_reconstruction=True))
    dump('GRID.json',configs());dump('PROTECTED.json',protected)


def registered():
    design();assert read(P/'GRID.json')==configs()
    plan=read(P/'PLAN.json');assert plan['design_sha256']==DESIGN_SHA256
    old.assert_hashes(plan['hashes']);old.assert_hashes(plan['source_hashes'])
    old.assert_hashes(plan['raw_source_hashes'])
    for study in components():assert study.features.calendar_provenance()==plan['calendar']


def assert_protected():old.assert_hashes(read(P/'PROTECTED.json'))

def source_spec(spec):
    if spec not in configs():
        raise ValueError('Only the five registered configurations are permitted')
    index=0 if spec['base']=='original65' else 1 if spec['base']=='incumbent66' else SOURCE_PRIMARY_ID
    value=upstream.configs()[index]
    if index==SOURCE_PRIMARY_ID:
        assert value==SOURCE_PRIMARY_SPEC
    return value

def _shape(spec,a,n):
    source_spec(spec)
    shape=np.asarray(a['baseline']).shape
    if (len(shape)!=2 or shape[1]==0 or isinstance(n,(bool,np.bool_))
            or not isinstance(n,(int,np.integer)) or not 0<=n<=shape[0]):
        raise ValueError('n must be a valid monthly prefix')
    return shape


def support_budget(a,n):
    """Conservative support reserve after old models and maximum episode use.

    The scope extension uses exactly the original finite/eligible/model mask.
    Episode clipping cannot consume more than its original reserved weight.
    """
    shape=np.asarray(a['baseline']).shape
    cw=adapter._core_weight(a['core_weight'],shape,n)
    calm=adapter._state(a,'state_calm10',shape[0],n)
    weak=adapter._state(a,'state_weak_narrow',shape[0],n)
    old=.025*calm[:,None]+.05*weak[:,None]
    episode=np.minimum(.05,np.maximum(0.,.10-old))
    extension=adapter._matrix(a['extension_available'],shape,n,'extension availability',boolean=True)
    common=adapter._matrix(a['forecast_available'],shape,n,'forecast availability',boolean=True)
    eligible=adapter._matrix(a['eligible'],shape,n,'eligibility',boolean=True)
    rank=adapter._matrix(a['rank_log_excess'],shape,n,'original log rank')
    active=(calm&~weak)[:,None]&extension&common&eligible&np.isfinite(rank)
    scope=.01*active
    remaining=1-cw-old-episode-scope
    if not np.isfinite(remaining).all() or (remaining<0.).any():
        raise ValueError('Invalid original auxiliary support reserve')
    return dict(core_weight=cw,prior_models=old,episode_reserve=episode,
        scope_extension=scope,remaining_original_support=remaining)


def factor_inputs(spec,a,n):
    _shape(spec,a,n)
    if spec['mode']=='ulcer_shift':return ulcer.rank_inputs(a,n)
    if spec['mode']=='bond_shift':return bond.effect_inputs(a,n)
    raise ValueError('Controls do not require a new factor')


def original_support(a,n):
    shape=np.asarray(a['baseline']).shape
    baseline=adapter._matrix(a['baseline'],shape,n,'baseline')
    core=adapter._matrix(a['core'],shape,n,'weighted core')
    cw=adapter._core_weight(a['core_weight'],shape,n)
    if not np.isfinite(baseline).all() or not np.isfinite(core).all():
        raise ValueError('Original baseline and weighted core must be finite')
    support=(baseline-core)/(1-cw)
    if not np.isfinite(support).all() or (np.abs(support)>.5+1e-12).any():
        raise ValueError('Original support must be finite within registered numeric bound')
    return support


def score_delta(spec,a,n):
    shape=_shape(spec,a,n);delta=np.zeros((n,shape[1]))
    if spec['mode']=='none':return delta
    factor,available=factor_inputs(spec,a,n)
    support=original_support(a,n)
    remaining=support_budget(a,n)['remaining_original_support']
    if (remaining[available]<.05).any():raise ValueError('Original support cannot fund fixed shift')
    # Explicit zero is essential even when original support has tolerated roundoff.
    active=available&(factor!=0.)
    delta[active]=.05*(np.clip(support[active]+factor[active],-.5,.5)-support[active])
    if not np.isfinite(delta).all() or (np.abs(delta)>.025+1e-12).any():
        raise FloatingPointError('Nonfinite or unbounded shift')
    return delta


def placement_audit(spec,a,n):
    factor,available=factor_inputs(spec,a,n);support=original_support(a,n)
    active=available&(factor!=0.);delta=score_delta(spec,a,n)
    after=support_budget(a,n)['remaining_original_support']-.05*available
    return dict(available_cells=int(available.sum()),zero_cells=int((available&(factor==0.)).sum()),
        missing_cells=int((~available).sum()),nonzero_factor_cells=int(active.sum()),
        modified_cells=int((delta!=0.).sum()),
        upper_saturation_cells=int((active&(support+factor>.5)).sum()),
        lower_saturation_cells=int((active&(support+factor<-.5)).sum()),
        support_min=float(support.min()),support_max=float(support.max()),
        support_roundoff_outside_exact_bound=int((np.abs(support)>.5).sum()),
        max_abs_delta=float(np.abs(delta).max()),min_remaining_original_support=float(after.min()),
        exact_zero_and_missing_delta=True,normalization_changed=False)


def modified_pre_score(spec,a,n):
    source=source_spec(spec)
    pre=upstream.modified_pre_score(source,a,n)
    delta=score_delta(spec,a,n)
    changed=delta!=0.
    prefix=pre[:n].copy();prefix[changed]+=delta[changed];pre[:n]=prefix
    return pre


def choices(spec,a,m,n,*,policy=None,boundary_selector=None):
    source=source_spec(spec);shape=_shape(spec,a,n)
    if spec['base']!='original65':
        policy=read(P/'INCUMBENT_POLICY.json') if policy is None else policy
        assert all(policy.get(k)==v for k,v in POLICY_FIELDS.items())
    if spec['mode']=='none':
        return upstream.choices(source,a,m,n,policy=policy,boundary_selector=boundary_selector)
    selector=ep.select_boundary if boundary_selector is None else boundary_selector
    return adapter.score_and_choose('incumbent66',0.,np.full(shape,np.nan),a,m,n,
        pre_score=modified_pre_score(spec,a,n),effects=a['effects'],
        episode_policy=policy,boundary_selector=selector)



def prepare():
    registered()
    assert not (P/'DATA.npz').exists() and not (P/'FROZEN.json').exists(),'Never overwrite prepared/frozen inputs'
    m,original,px,op,policy,bundles,raw,calendar=verify_source()
    n=len(m['dates']);shape=np.asarray(original['baseline']).shape
    assert n==122 and shape==(122,49) and m['dates'][0]=='2016-08-17'
    assert m['ends'][87]=='2023-11-30' and m['ends'][-1]=='2026-09-24'
    plan=read(P/'PLAN.json');assert raw==plan['raw_source_hashes'] and calendar==plan['calendar']
    a={key:value.copy() for key,value in original.items()}
    for data in bundles:
        for key,value in data.items():
            if key not in a:a[key]=value.copy()
    validate_cached_features(a,shape)
    controls={}
    for spec in configs()[:3]:
        ident=spec['id'];source=source_spec(spec)
        expected=upstream.choices(source,original,m,n,policy=policy)
        actual=choices(spec,a,m,n,policy=policy)
        assert actual[0]==expected[0];np.testing.assert_array_equal(actual[1],expected[1])
        orders=rp.make_orders(*actual,m)
        assert orders==rp.make_orders(*expected,m)
        assert orders==read(upstream.P/('original65_ORDERS.json','incumbent66_ORDERS.json','primary_ORDERS.json')[ident])
        name=('original65','incumbent66','incumbent_scope67')[ident]
        for study,data in zip(components(),bundles):
            paired=study.choices(study.configs()[ident],data,m,n,policy=policy)
            assert paired[0]==actual[0];np.testing.assert_array_equal(paired[1],actual[1])
            assert orders==read(study.P/(name+'_ORDERS.json'))
        controls[ident]=actual
    audits={}
    original_pre=upstream.modified_pre_score(SOURCE_PRIMARY_SPEC,a,n)
    for spec in configs()[3:]:
        factor,available=factor_inputs(spec,a,n)
        prefix='ulcer_' if spec['mode']=='ulcer_shift' else 'bond_'
        key=prefix+('rank' if prefix=='ulcer_' else 'effect')
        for valid,value in ((np.zeros(shape,bool),np.full(shape,np.nan)),(np.ones(shape,bool),np.zeros(shape))):
            neutral=dict(a,**{prefix+'available':valid,key:value})
            got=choices(spec,neutral,m,n,policy=policy)
            assert got[0]==controls[2][0];np.testing.assert_array_equal(got[1],controls[2][1])
            assert rp.make_orders(*got,m)==rp.make_orders(*controls[2],m)
            np.testing.assert_array_equal(score_delta(spec,neutral,n),np.zeros(shape))
        delta=score_delta(spec,a,n);inactive=~available|(factor==0.)
        np.testing.assert_array_equal(delta[inactive],np.zeros(inactive.sum()))
        np.testing.assert_array_equal(modified_pre_score(spec,a,n)[inactive],original_pre[inactive])
        full=choices(spec,a,m,n,policy=policy);prefix_choice=choices(spec,a,m,TRAIN,policy=policy)
        assert prefix_choice[0]==full[0][:TRAIN]
        np.testing.assert_array_equal(prefix_choice[1],full[1][:TRAIN])
        audits[diagnostic_name(spec)]=placement_audit(spec,a,n)
    np.savez_compressed(P/'DATA.npz',**a)
    dump('META.json',m);dump('INCUMBENT_POLICY.json',policy)
    dump('FEATURE_AUDIT.json',dict(component_sources=[str(study.P) for study in components()],
        frozen_cache_reuse=True,no_feature_reconstruction=True,no_rerank=True,
        independent_availability=True,calendar=calendar,raw_source_hashes=raw,
        shared_arrays_dtype_values_metadata_execution_prices_policy_exact=True))
    dump('PLACEMENT_AUDIT.json',dict(arms=audits,fixed_aux_slot=.05,
        formula='.05*(clip(original_support+F,-.5,.5)-original_support)',
        original_support='(original_baseline-original_weighted_core)/(1-original_core_weight)',
        original_core_coefficients=sorted(np.unique(a['core_weight']).tolist()),
        zero_and_missing_exact_identity=True,original_episode_budget_unchanged=True,
        original_support_payload_exact=True,modified_pre_drives_top2_and_boundary=True,
        inactive_cell_final_membership_may_change=True,no_new_model_or_episode_budget=True))
    dump('PREPARE_VALIDATION.json',dict(three_controls_scores_choices_orders_exact=True,
        both_components_shared_source_exact=True,no_component_performance_filter=True,
        original_factors_and_masks_exact=True,all_zero_all_missing_exact_source=True,
        candidate_prefix_exact=True,source_freeze_verified=True,
        core_coefficients_unchanged=True,eligibility_unchanged=True,no_global_rerank=True))
    hashes=upstream.upstream.merge_hashes(read(upstream.P/'INPUT_HASHES.json'),
        *(read(study.P/'INPUT_HASHES.json') for study in components()))
    for f in list(implementations())+list(source_files())+[P/'DESIGN.json',P/'DESIGN.sha256']:
        digest=sha(f);assert str(f) not in hashes or hashes[str(f)]==digest,str(f);hashes[str(f)]=digest
    dump('INPUT_HASHES.json',hashes)
    dump('CACHE_HASHES.json',{f:sha(P/f) for f in ('DATA.npz','META.json','INCUMBENT_POLICY.json',
        'FEATURE_AUDIT.json','PLACEMENT_AUDIT.json','PREPARE_VALIDATION.json','INPUT_HASHES.json')})
    print('Prepared five fixed auxiliary shifts; frozen factors and three controls exact',flush=True)


def load():
    registered();verify_source()
    old.assert_hashes(read(P/'INPUT_HASHES.json'));old.assert_hashes(read(P/'CACHE_HASHES.json'),P)
    m,_,px,op=upstream.load();assert m==read(P/'META.json')
    with np.load(P/'DATA.npz') as data:a={key:data[key].copy() for key in data.files}
    validate_cached_features(a,np.asarray(a['baseline']).shape)
    return m,a,px,op


def admitted(z, incumbent):
    return (z['training']['cagr'] >= incumbent['training']['cagr']-1e-12
        and z['training']['sharpe'] >= incumbent['training']['sharpe']-.05
        and z['training']['max_drawdown'] >= incumbent['training']['max_drawdown']-.03
        and z['early']['cagr'] >= incumbent['early']['cagr']-.025
        and z['recent']['cagr'] >= incumbent['recent']['cagr']-.025)


def select(rows):
    assert [z['spec'] for z in rows] == configs()
    incumbent = rows[2]
    for z in rows:
        z['admitted'] = admitted(z,incumbent)
    qualified = [incumbent]+[z for z in rows[3:] if z['admitted']]
    best = max(z['training']['cagr'] for z in qualified)
    primary = min((z for z in qualified if best-z['training']['cagr'] <= 1e-12),
                  key=lambda z:z['spec']['id'])
    selected = dict(primary=primary,original65=rows[0],incumbent66=rows[1],incumbent_scope67=incumbent)
    selected.update({diagnostic_name(z['spec']):z for z in rows[3:]})
    return selected

def train():
    registered()
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
        control.extend(returns[2][bounds[n]:bounds[end]])
        folds.append(dict(train_end=m['ends'][n-1],test_end=m['ends'][end-1],spec=winner['spec']))
    dump('WALKFORWARD.json',dict(folds=folds,pristine_oos=False,
        deployment='Stateless monthly selection; each fold uses only its selected fixed rule, with no prior-holding dependency.',
        metrics=rp.stats_rr(np.array(stitched),m['dates'][36],m['ends'][87]),
        incumbent=rp.stats_rr(np.array(control),m['dates'][36],m['ends'][87])))
    dump('TRAIN_TRIALS.json',rows)
    np.savez_compressed(P/'TRAIN_RETURNS.npz',returns=np.stack(returns),bounds=np.array(bounds))
    files=('DESIGN.json','DESIGN.sha256','PLAN.json','GRID.json','DATA.npz','META.json','INCUMBENT_POLICY.json',
           'FEATURE_AUDIT.json','PLACEMENT_AUDIT.json','PREPARE_VALIDATION.json','INPUT_HASHES.json',
           'CACHE_HASHES.json','PROTECTED.json','study.py','test_study.py')
    dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),selected=selected,
        trials=5,new_admitted=sum(int(z['admitted']) for z in rows[3:]),
        primary_source=selected['primary']['spec'],
        zero_and_missing_exact_identity=True,component_performance_used_as_algorithm_input=False,
        hashes={f:sha(P/f) for f in files},pristine_oos=False,author_labels_used=False,
        diagnostics_may_be_unqualified=True,both_diagnostics_frozen=True,cagr_tie_tolerance=1e-12))
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
            ch=[[m['symbols'][i].index(s) for s in row['symbols']] for i,row in enumerate(oo)]
            rr=rp.fast_returns(ch,blocks,valid)
            np.testing.assert_allclose(eq,np.r_[1.,np.cumprod(1+rr)]*100000,rtol=1e-11,atol=1e-5)
            assert abs(common['slices']['training2016_2023']['cagr']-frozen['selected'][name]['training']['cagr'])<1e-10
        results[name]=dict(common,selection=frozen['selected'].get(name))
        for suffix,value in (('ORDERS',oo),('DAILY',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()]),('TRADES',tr),('HOLDINGS',hold)):
            dump(name+'_'+suffix+'.json',value)
        print(name,round(common['metrics']['cagr']*100,3),round(common['metrics']['sharpe'],3),flush=True)
    for name,filename in (('original65','original65_ORDERS.json'),
            ('incumbent66','incumbent66_ORDERS.json'),('incumbent_scope67','primary_ORDERS.json')):
        assert orders[name]==read(upstream.P/filename), 'Source control order drift: '+name
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
    for name,reference in [(diagnostic_name(z),'incumbent_scope67') for z in configs()[3:]]:
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
    # The inherited large protection inventory is audited once, after all phases.
    assert_protected()
    dump('RESULTS.json',results)
    dump('VALIDATION.json',dict(original_controls_exact=True,independent_fixed_unit_replay=True,
        training_freeze_parity=True,protected_files=len(read(P/'PROTECTED.json')),protected_changes=[],
        observer_accounts_changed=False,pristine_oos=False,fixed_configs=5,
        core_coefficients_unchanged=True,aux_support_shift_slot=.05,zero_and_missing_exact_identity=True,nominal_episode_budget=.05,both_diagnostics_frozen=True,
        modified_pre_drives_top2=True,original_support_payload_exact=True,true_episode_boundary_rules=True,
        no_global_rerank=True,no_intramonth_transactions=True,no_refit=True,
        component_performance_used_as_algorithm_input=False,original_factors_and_masks_exact=True,
        primary_target70_reached=results['primary']['metrics']['cagr']>=.70))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('initialize','prepare','train','evaluate'))
    globals()[parser.parse_args().phase]()
