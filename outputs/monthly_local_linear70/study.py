"""Matched same-security residual constant/linear corrections to annual Ridge.

Unchanged pooled log targets and annual cohort remain the source model.
Only one quarter of its existing scoring slots can change; no new budget.
"""
from pathlib import Path
from datetime import datetime,timezone
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
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


scope=module('local_linear70_scope',O/'monthly_model_scope70/study.py')
target=scope.upstream
old,rp,ex,arch,ep,read,sha,adapter=(scope.old,scope.rp,scope.ex,scope.arch,scope.ep,scope.read,scope.sha,scope.adapter)
calendar_labels=module('local_linear70_calendar',O/'monthly_horizon70_research/labels.py')
model=module('local_linear70_model',P/'model.py')
TRAIN=88
PARITY_ATOL=1e-12
DESIGN_SHA256='85f9b3d93f7c79297a4a6fc906af7f661f2a2718a54669830a785a1808603159'
POLICY_FIELDS=dict(scope.POLICY_FIELDS)
SOURCE_SCOPE_SPEC=dict(id=3,base='incumbent67',scope='calm_nonweak',budget=.01)
ARMS=('local_constant','local_linear')


def dump(name,value):
    (P/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else x)+'\n')


def configs():
    return [dict(id=0,base='original65',mode='none'),dict(id=1,base='incumbent66',mode='none'),
        dict(id=2,base='incumbent_scope67',mode='none'),
        dict(id=3,base='incumbent_scope67',mode='local_constant'),
        dict(id=4,base='incumbent_scope67',mode='local_linear')]


def diagnostic_name(spec):
    if spec not in configs()[3:]:raise ValueError('Only fixed paired candidates have diagnostic names')
    return spec['mode']


def implementations():
    return tuple(sorted(set(scope.implementations())|set(model.dependency_files())|{Path(calendar_labels.__file__),calendar_labels.CALENDAR_PATH}|{
        P/'study.py',P/'test_study.py',P/'model.py',P/'test_model.py'}))


def source_files():
    files=[scope.P/name for name in ('DESIGN.json','PLAN.json','GRID.json','INPUT_HASHES.json','CACHE_HASHES.json',
        'FROZEN.json','FROZEN.sha256','DATA.npz','META.json','INCUMBENT_POLICY.json','PROTECTED.json',
        'primary_ORDERS.json','original65_ORDERS.json','incumbent66_ORDERS.json')]
    files += [target.P/name for name in ('TRAIN_INPUTS.npz','PLAN.json','INPUT_HASHES.json','CACHE_HASHES.json',
        'FROZEN.json','FROZEN.sha256','DATA.npz','META.json','LABEL_AUDIT.json','MODEL_AUDIT.json')]
    files += [target.old.P/name for name in ('DATA.npz','META.json','INPUT_HASHES.json')]
    files += [arch.P/'META.json',arch.P/'INPUT_HASHES.json']
    files += [Path(path) for path in calendar_provenance()['source_hashes']]
    return tuple(sorted(set(files)))


def design():
    assert sha(P/'DESIGN.json')==DESIGN_SHA256,'Frozen local-linear design changed'
    assert (P/'DESIGN.sha256').read_text().strip()==DESIGN_SHA256
    fixed=read(P/'DESIGN.json');assert fixed['configs']==configs()
    return fixed


def verify_source():
    scope.registered();scope.verify_source()
    assert scope.configs()[3]==SOURCE_SCOPE_SPEC
    assert sha(scope.P/'FROZEN.json')==(scope.P/'FROZEN.sha256').read_text().strip()
    frozen=read(scope.P/'FROZEN.json')
    assert frozen['selected']['primary']['spec']==SOURCE_SCOPE_SPEC
    old.assert_hashes(frozen['hashes'],scope.P)
    for study in (scope,target):
        old.assert_hashes(read(study.P/'INPUT_HASHES.json'))
        old.assert_hashes(read(study.P/'CACHE_HASHES.json'),study.P)


def raw_source_hashes(info):
    hashes=dict(info['source_hashes']);loader=O/'monthly_stability_monitor/inputs.py'
    assert hashes.get(str(loader))==sha(loader) and len(hashes)==64
    source=read(target.P/'INPUT_HASHES.json')
    assert all(source.get(path)==digest for path,digest in hashes.items()),'Original raw input changed'
    return hashes


def initialize():
    assert not (P/'PLAN.json').exists(),'Never overwrite registration'
    fixed=design();verify_source()
    _,info=arch.old.panel_load(read(scope.P/'META.json')['etfs'])
    protected=read(scope.P/'PROTECTED.json');assert len(protected)==21436
    dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        design_sha256=DESIGN_SHA256,design=fixed,
        raw_source_hashes=raw_source_hashes(info),calendar=calendar_provenance(),
        parity='Independently call original four-target model; source_log raw atol1e-12 rtol0, exact availability/ranks/three controls.',
        local_cohort='Same-security subset of original annual mature global cohort; both arms retain pooled raw for fewer than18 own months.',
        hashes={str(f):sha(f) for f in implementations()},source_hashes={str(f):sha(f) for f in source_files()}))
    dump('GRID.json',configs());dump('PROTECTED.json',protected)


def registered():
    design();assert read(P/'GRID.json')==configs()
    plan=read(P/'PLAN.json');assert plan['design_sha256']==DESIGN_SHA256
    old.assert_hashes(plan['hashes']);old.assert_hashes(plan['source_hashes'])
    old.assert_hashes(plan['raw_source_hashes'])
    assert calendar_provenance()==plan['calendar']


def assert_protected():old.assert_hashes(read(P/'PROTECTED.json'))


def source_spec(spec):
    if spec not in configs():raise ValueError('Only five fixed configurations permitted')
    return scope.configs()[0 if spec['id']==0 else 1 if spec['id']==1 else 3]


def _shape(a,n):
    shape=np.asarray(a['baseline']).shape
    if (len(shape)!=2 or not shape[1] or isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer))
            or not 0<=n<=shape[0]):raise ValueError('n must be a valid monthly prefix')
    return shape


def original_rank_inputs(a,n):
    shape=_shape(a,n)
    original=adapter._matrix(a['rank_log_excess'],shape,n,'original logrank')
    if not np.isfinite(original).all() or (np.abs(original)>.5+1e-12).any():
        raise ValueError('Original ranks must be finite and bounded')
    available=adapter._matrix(a['forecast_available'],shape,n,'original availability',boolean=True)
    eligible=adapter._matrix(a['eligible'],shape,n,'eligibility',boolean=True)
    return original,available&eligible


def source_ranked_forecast(raw,a):
    """Full source parity check must not hide availability or ranking expansion."""
    n=len(a['baseline']);shape=_shape(a,n)
    raw=adapter._matrix(raw,shape,n,'source raw')
    original,use=original_rank_inputs(a,n)
    if not np.isfinite(raw[use]).all():raise ValueError('Source available predictions must be finite')
    out=original.copy()
    for i in range(n):
        if use[i].any():
            rank=ep.old.prior.ranks(pd.Series(raw[i]).where(use[i]))
            out[i,use[i]]=rank[use[i]]
    return out


def ranked_forecasts(raw,available,a,n):
    """Rank both arms on complete original U, or keep both entire old rows.

    NaN/unavailable are legitimate missing forecasts; infinity is a numerical
    error. Production models separately reject all nonfinite fitted outputs.
    No own-history, slot-state, or future-label mask enters U.
    """
    shape=_shape(a,n)
    if set(raw)!=set(ARMS):raise ValueError('Exactly two paired raw forecast matrices required')
    values={key:adapter._matrix(raw[key],shape,n,key) for key in ARMS}
    if any(np.isinf(value).any() for value in values.values()):raise FloatingPointError('Infinite forecast')
    available=adapter._matrix(available,shape,n,'pooled availability',boolean=True)
    original,use=original_rank_inputs(a,n)
    source=adapter._matrix(a['raw_source_log'],shape,n,'replayed source raw')
    if not np.isfinite(source[use]).all():raise ValueError('Source available predictions must be finite')
    ready=np.zeros(n,bool);ranks={key:original.copy() for key in ARMS}
    for i in range(n):
        ready[i]=bool(use[i].any() and available[i,use[i]].all()
            and all(np.isfinite(value[i,use[i]]).all() for value in values.values()))
        if ready[i]:
            for key in ARMS:
                if np.array_equal(values[key][i,use[i]],source[i,use[i]]):continue
                rank=ep.old.prior.ranks(pd.Series(values[key][i]).where(use[i]))
                ranks[key][i,use[i]]=rank[use[i]]
    return ranks,ready


def score_delta(spec,a,n):
    source_spec(spec);shape=_shape(a,n);delta=np.zeros((n,shape[1]))
    if spec['mode']=='none':return delta
    ranks,ready=ranked_forecasts({key:a['raw_'+key] for key in ARMS},a['model_available'],a,n)
    current,use=original_rank_inputs(a,n)
    rank=ranks[spec['mode']]
    weak=adapter._state(a,'state_weak_narrow',shape[0],n)
    calm=adapter._state(a,'state_calm10',shape[0],n)
    extension=adapter._matrix(a['extension_available'],shape,n,'extension availability',boolean=True)
    coefficient=.25*(.05*weak[:,None]+.01*(calm&~weak)[:,None]*extension)
    active=use&ready[:,None]
    delta[active]=coefficient[active]*(rank[active]-current[active])
    return delta


def modified_pre_score(spec,a,n):
    source=source_spec(spec);_shape(a,n)
    pre=scope.modified_pre_score(source,a,n)
    delta=score_delta(spec,a,n);changed=delta!=0.
    view=pre[:n].copy();view[changed]+=delta[changed];pre[:n]=view
    return pre


def choices(spec,a,m,n,*,policy=None,boundary_selector=None):
    source=source_spec(spec);shape=_shape(a,n)
    if spec['base']!='original65':
        policy=read(P/'INCUMBENT_POLICY.json') if policy is None else policy
        assert all(policy.get(k)==v for k,v in POLICY_FIELDS.items())
    if spec['mode']=='none':
        return scope.choices(source,a,m,n,policy=policy,boundary_selector=boundary_selector)
    return adapter.score_and_choose('incumbent66',0.,np.full(shape,np.nan),a,m,n,
        pre_score=modified_pre_score(spec,a,n),effects=a['effects'],episode_policy=policy,
        boundary_selector=ep.select_boundary if boundary_selector is None else boundary_selector)


def validate_identity(meta,cached,original,architecture_meta):
    """Verify original row identity and untouched cached data, without labels."""
    assert meta['training']==architecture_meta['training']
    assert meta['monthly']==architecture_meta['monthly']
    for key in target.CACHED_KEYS:
        assert np.asarray(cached[key]).dtype==np.asarray(original[key]).dtype,'Changed cache dtype: '+key
        np.testing.assert_array_equal(cached[key],original[key])
    tm=meta['training']
    assert len(tm['dates'])==len(tm['ends'])==len(tm['symbols'])
    assert all(len(ss)==len(set(ss)) and all(isinstance(s,str) and s for s in ss) for ss in tm['symbols'])
    widths=np.array([len(ss) for ss in tm['symbols']],dtype=int)
    offsets=np.r_[0,np.cumsum(widths)]
    np.testing.assert_array_equal(cached['offsets'],offsets)
    for key in ('dates','ends'):
        np.testing.assert_array_equal(np.asarray(cached[key],dtype='datetime64[D]'),
            np.repeat(np.array(tm[key],dtype='datetime64[D]'),widths))
    symbols=np.array([s for ss in tm['symbols'] for s in ss])
    assert cached['train_plain'].shape==(len(symbols),10)
    for key in ('train_eligible','train_good'):
        assert cached[key].shape==(len(symbols),) and cached[key].dtype.kind=='b'
    m=meta['monthly'];widths=np.array([len(ss) for ss in m['symbols']])
    np.testing.assert_array_equal(np.asarray(cached['event_dates'],dtype='datetime64[D]'),
        np.repeat(np.array(m['dates'],dtype='datetime64[D]'),widths))
    assert cached['event_plain'].shape==(widths.sum(),10)
    return dict(training_months=len(offsets)-1,rows=len(symbols),eligible_rows=int(cached['train_eligible'].sum()),
        exact_all9_source_arrays=True,historical_metadata_equal_architecture=True,
        explicit_symbols_offsets_dates_ends=True,no_price_or_label_reconstruction=True)


def verify_panel_parity(panel,px,op):
    """Execution levels may have one fixed historical scale per security."""
    for frame in (px,op):
        if not frame.index.equals(px.index) or not frame.columns.equals(px.columns):
            raise ValueError('Source execution price axes differ')
    if not px.index.isin(panel['adj'].index).all() or not px.columns.isin(panel['adj'].columns).all():
        raise ValueError('Raw panel does not cover the source execution axes')
    adj=panel['adj'].reindex(index=px.index,columns=px.columns)
    opening=panel['open'].reindex(index=px.index,columns=px.columns)
    close=panel['close'].reindex(index=px.index,columns=px.columns)
    scales={}
    all_missing=[]
    for symbol in px.columns:
        raw,source=adj[symbol].to_numpy(float),px[symbol].to_numpy(float)
        np.testing.assert_array_equal(np.isfinite(raw),np.isfinite(source))
        np.testing.assert_array_equal(np.isfinite(raw)&(raw>0.),np.isfinite(source)&(source>0.))
        if ((np.isfinite(raw)&(raw<=0.)).any() or (np.isfinite(source)&(source<=0.)).any()):
            raise ValueError('Nonpositive execution/raw price in parity check: '+symbol)
        valid=np.isfinite(raw)&(raw>0.)&np.isfinite(source)&(source>0.)
        if not valid.any():
            all_missing.append(symbol)
            continue
        scale=float(source[valid][0]/raw[valid][0])
        np.testing.assert_allclose(source[valid],raw[valid]*scale,rtol=1e-10,atol=1e-10)
        scales[symbol]=scale
    expected=px*opening/close
    np.testing.assert_allclose(op.to_numpy(float),expected.to_numpy(float),
        rtol=1e-10,atol=1e-10,equal_nan=True)
    return dict(adjusted_levels_constant_scale=True,adjusted_opening_parity=True,
        missing_price_masks_exact=True,positive_price_masks_exact=True,
        symbols=len(scales),scales=scales,all_missing_symbols=all_missing)


def calendar_provenance():
    provenance=calendar_labels._CAL.provenance()
    return dict(provenance=provenance,source_hashes=calendar_labels._source_hashes(provenance))


def validate_forecasts(forecast,rows):
    if set(forecast['raw'])!=set(('source_log',*ARMS)):raise ValueError('Wrong forecast keys')
    for name in ('available','local_available'):
        value=np.asarray(forecast[name])
        if value.shape!=(rows,) or value.dtype.kind!='b':raise ValueError('Invalid '+name)
    if (forecast['local_available']&~forecast['available']).any():raise ValueError('Local history expands source availability')
    for key,value in forecast['raw'].items():
        value=np.asarray(value)
        if value.shape!=(rows,):raise ValueError('Forecast row shape differs')
        if np.isinf(value).any() or not np.array_equal(np.isfinite(value),forecast['available']):
            raise FloatingPointError('Model outputs must be finite exactly where source available: '+key)
    cold=forecast['available']&~forecast['local_available']
    for key in ARMS:
        np.testing.assert_array_equal(forecast['raw'][key][cold],forecast['raw']['source_log'][cold])


def prepare():
    registered()
    assert not (P/'DATA.npz').exists() and not (P/'FROZEN.json').exists(),'Never overwrite prepared/frozen data'
    verify_source();m,a,px,op=scope.load();policy=read(scope.P/'INCUMBENT_POLICY.json')
    with np.load(target.P/'TRAIN_INPUTS.npz') as stored:cached={key:stored[key].copy() for key in stored.files}
    with np.load(target.old.P/'DATA.npz') as stored:original={key:stored[key].copy() for key in target.CACHED_KEYS}
    meta=read(target.old.P/'META.json')
    identity=validate_identity(meta,cached,original,read(arch.P/'META.json'))
    target.validate_axes(m,meta,cached,a['eligible'])
    assert cached['train_plain'].shape==(8820,10) and cached['event_plain'].shape==(5978,10)
    assert a['baseline'].shape==(122,49)
    symbols=np.array([symbol for pool in meta['training']['symbols'] for symbol in pool])
    panel,info=arch.old.panel_load(m['etfs']);raw=raw_source_hashes(info)
    assert raw==read(P/'PLAN.json')['raw_source_hashes']
    parity=verify_panel_parity(panel,px,op)
    source_targets={name:cached['target_'+name] for name in target.TARGETS}
    event_symbols=np.array([symbol for pool in m['symbols'] for symbol in pool])
    event_good=a['eligible'].reshape(-1)&np.isfinite(cached['event_plain']).all(axis=1)
    forecast=model.annual_forecasts(cached['dates'],cached['ends'],symbols,cached['train_plain'],source_targets,
        cached['common_good'],cached['event_dates'],event_symbols,cached['event_plain'],event_good)
    validate_forecasts(forecast,len(cached['event_dates']))
    shape=a['baseline'].shape
    source_available=forecast['available'].reshape(shape)
    np.testing.assert_array_equal(source_available,a['forecast_available'])
    source_raw=forecast['raw']['source_log'].reshape(shape)
    np.testing.assert_allclose(source_raw,a['raw_log_excess'],rtol=0.,atol=PARITY_ATOL,equal_nan=True)
    np.testing.assert_array_equal(source_ranked_forecast(source_raw,a),a['rank_log_excess'])
    a['raw_source_log']=source_raw;a['model_available']=forecast['available'].reshape(shape)
    a['local_available']=forecast['local_available'].reshape(shape)
    for key in ARMS:a['raw_'+key]=forecast['raw'][key].reshape(shape)
    ranks,ready=ranked_forecasts({key:a['raw_'+key] for key in ARMS},a['model_available'],a,len(m['dates']))
    for key in ARMS:a['rank_'+key]=ranks[key]
    a['paired_month_ready']=ready
    controls={}
    for ident,sourceid in ((0,0),(1,1),(2,3)):
        got=choices(configs()[ident],a,m,len(m['dates']),policy=policy)
        expected=scope.choices(scope.configs()[sourceid],a,m,len(m['dates']),policy=policy)
        assert got[0]==expected[0];np.testing.assert_array_equal(got[1],expected[1])
        filename=('original65_ORDERS.json','incumbent66_ORDERS.json','primary_ORDERS.json')[ident]
        assert rp.make_orders(*got,m)==read(scope.P/filename)
        controls[ident]=got
    fallback=dict(a,model_available=np.zeros(shape,bool),**{'raw_'+key:np.full(shape,np.nan) for key in ARMS})
    for spec in configs()[3:]:
        got=choices(spec,fallback,m,len(m['dates']),policy=policy)
        assert got[0]==controls[2][0];np.testing.assert_array_equal(got[1],controls[2][1])
        assert rp.make_orders(*got,m)==rp.make_orders(*controls[2],m)
        full=choices(spec,a,m,len(m['dates']),policy=policy);prefix=choices(spec,a,m,TRAIN,policy=policy)
        assert prefix[0]==full[0][:TRAIN];np.testing.assert_array_equal(prefix[1],full[1][:TRAIN])
    np.savez_compressed(P/'ROW_IDENTITIES.npz',dates=cached['dates'],ends=cached['ends'],symbols=symbols,
        event_dates=cached['event_dates'],event_symbols=event_symbols,offsets=cached['offsets'])
    np.savez_compressed(P/'DATA.npz',**a)
    dump('META.json',m);dump('INCUMBENT_POLICY.json',policy)
    dump('IDENTITY_AUDIT.json',identity);dump('MODEL_AUDIT.json',forecast['audit'])
    original_use=a['forecast_available']&a['eligible']
    dump('RANK_AUDIT.json',dict(population='original forecast_available & original eligible for all query months',
        original_population_cells=int(original_use.sum()),paired_ready_months=int(ready.sum()),
        both_fallback_months=int((original_use.any(axis=1)&~ready).sum()),
        empty_original_population_months=int((~original_use.any(axis=1)).sum()),
        local_available_cells=int((a['local_available']&original_use).sum()),
        source_equal_months={key:int(sum(bool(use.any() and np.array_equal(a['raw_'+key][i,use],source_raw[i,use]))
            for i,use in enumerate(original_use))) for key in ARMS},
        source_equal_raw_reuses_cached_rank=True,no_own_history_gate=True,no_ranking_population_shrink=True,two_arms_common_month_gate=True))
    dump('PREPARE_VALIDATION.json',dict(all_three_source_controls_scores_orders_exact=True,
        source_model_internal_only=True,source_log_raw_atol=PARITY_ATOL,source_availability_rank_exact=True,
        source_label_cache_exact=True,global_cohort_unchanged=True,local_constant_is_matched_control=True,
        both_missing_exact_current_scope=True,own_history_not_required=True,all_candidates_prefix_exact=True,
        no_new_ml_budget=True,existing_slot_replacement_fraction=.25,original_support_payload_exact=True,raw_panel_parity=parity))
    hashes=target.merge_hashes(read(scope.P/'INPUT_HASHES.json'),read(target.P/'INPUT_HASHES.json'),
        info['source_hashes'],calendar_provenance()['source_hashes'])
    for f in list(implementations())+list(source_files())+[P/'DESIGN.json',P/'DESIGN.sha256']:
        digest=sha(f);assert str(f) not in hashes or hashes[str(f)]==digest,str(f);hashes[str(f)]=digest
    dump('INPUT_HASHES.json',hashes)
    dump('CACHE_HASHES.json',{f:sha(P/f) for f in ('DATA.npz','ROW_IDENTITIES.npz','META.json','INCUMBENT_POLICY.json',
        'IDENTITY_AUDIT.json','MODEL_AUDIT.json','RANK_AUDIT.json','PREPARE_VALIDATION.json','INPUT_HASHES.json')})
    print('Prepared five fixed local-residual configurations; original source parity and two matched ranks verified',flush=True)


def load():
    registered();verify_source()
    old.assert_hashes(read(P/'INPUT_HASHES.json'));old.assert_hashes(read(P/'CACHE_HASHES.json'),P)
    m,_,px,op=scope.load();assert m==read(P/'META.json')
    with np.load(P/'DATA.npz') as data:a={key:data[key].copy() for key in data.files}
    ranks,ready=ranked_forecasts({key:a['raw_'+key] for key in ARMS},a['model_available'],a,len(m['dates']))
    for key in ARMS:np.testing.assert_array_equal(ranks[key],a['rank_'+key])
    np.testing.assert_array_equal(ready,a['paired_month_ready'])
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
           'IDENTITY_AUDIT.json','MODEL_AUDIT.json','RANK_AUDIT.json','ROW_IDENTITIES.npz','PREPARE_VALIDATION.json','INPUT_HASHES.json',
           'CACHE_HASHES.json','PROTECTED.json','study.py','model.py','test_study.py','test_model.py')
    dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),selected=selected,
        trials=5,new_admitted=sum(int(z['admitted']) for z in rows[3:]),
        primary_source=selected['primary']['spec'],
        source_log_internal_parity_only=True,global_cohort_unchanged=True,
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
        assert orders[name]==read(scope.P/filename), 'Source control order drift: '+name
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
    for name,reference in [(diagnostic_name(z),'incumbent_scope67') for z in configs()[3:]]+[('local_linear','local_constant')]:
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
        core_coefficients_unchanged=True,existing_weak_slot=.05,existing_calm_nonweak_slot=.01,nominal_episode_budget=.05,both_diagnostics_frozen=True,
        modified_pre_drives_top2=True,original_support_payload_exact=True,true_episode_boundary_rules=True,
        no_global_rerank=True,no_intramonth_transactions=True,unchanged_plain10_and_annual_clock=True,
        source_model_parity=True,global_cohort_unchanged=True,matched_pair_attribution=True,slot_replacement_fraction=.25,
        primary_target70_reached=results['primary']['metrics']['cagr']>=.70))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('initialize','prepare','train','evaluate'))
    globals()[parser.parse_args().phase]()
