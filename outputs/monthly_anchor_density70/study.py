"""Matched monthly/weekly training anchors under the fixed annual pooled fit.

Four immutable controls, two fixed strengths starting at original scope pre.
The newer local-linear quarter control is a benchmark, never a stacked input.
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


scope=module('anchor_density70_scope',O/'monthly_model_scope70/study.py')
target=scope.upstream
old,rp,ex,arch,ep,read,sha,adapter=(scope.old,scope.rp,scope.ex,scope.arch,scope.ep,scope.read,scope.sha,scope.adapter)
local=module('density70_local_control',O/'monthly_local_linear70/study.py')
model=module('anchor_density70_model',P/'model.py')
features=module('anchor_density70_features',P/'features.py')
labels=module('anchor_density70_labels',P/'labels.py')
ARMS=('month21','union21')
TRAIN=88
PARITY_ATOL=1e-12
DESIGN_SHA256='4117f9fd26bd8e5c760f345e02668c748e18d63cc9c2d94d464c7e960282e0df'
POLICY_FIELDS=dict(scope.POLICY_FIELDS)
SOURCE_SCOPE_SPEC=dict(id=3,base='incumbent67',scope='calm_nonweak',budget=.01)
LOCAL_SPEC=dict(id=4,base='incumbent_scope67',mode='local_linear')


def dump(name,value):
    (P/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else x)+'\n')


def configs():
    return [dict(id=0,base='original65',mode='none'),dict(id=1,base='incumbent66',mode='none'),
        dict(id=2,base='incumbent_scope67',mode='none'),dict(id=3,base='incumbent_local67',mode='none'),
        dict(id=4,base='incumbent_scope67',mode='month21'),dict(id=5,base='incumbent_scope67',mode='union21')]


def diagnostic_name(spec):
    if spec not in configs()[4:]:raise ValueError('Only the two fixed density candidates have diagnostic names')
    return spec['mode']


def implementations():
    return tuple(sorted(set(local.implementations())|set(model.dependency_files())|set(features.dependency_files())|set(labels.dependency_files())|{
        P/'study.py',P/'test_study.py',P/'model.py',P/'test_model.py',P/'features.py',P/'test_features.py',P/'labels.py',P/'test_labels.py'}))


def calendar_provenance():return features.calendar_provenance()


def source_files():
    files=set(local.source_files())
    files.update(local.P/name for name in ('DESIGN.json','DESIGN.sha256','PLAN.json','GRID.json','INPUT_HASHES.json','CACHE_HASHES.json',
        'FROZEN.json','FROZEN.sha256','DATA.npz','META.json','INCUMBENT_POLICY.json','PROTECTED.json',
        'ROW_IDENTITIES.npz','IDENTITY_AUDIT.json','MODEL_AUDIT.json','RANK_AUDIT.json','PREPARE_VALIDATION.json','VALIDATION.json',
        'local_linear_ORDERS.json','original65_ORDERS.json','incumbent66_ORDERS.json','incumbent_scope67_ORDERS.json'))
    files.add(features.LEGACY_PATH)
    files.update(Path(path) for path in calendar_provenance()['source_hashes'])
    return tuple(sorted(files))


def design():
    assert sha(P/'DESIGN.json')==DESIGN_SHA256,'Frozen matched anchor-density design changed'
    assert (P/'DESIGN.sha256').read_text().strip()==DESIGN_SHA256
    fixed=read(P/'DESIGN.json');assert fixed['grid']==configs()
    return fixed


def verify_source():
    local.registered();local.verify_source()
    assert local.configs()[4]==LOCAL_SPEC
    assert sha(local.P/'FROZEN.json')==(local.P/'FROZEN.sha256').read_text().strip()
    frozen=read(local.P/'FROZEN.json')
    assert frozen['selected']['local_linear']['spec']==LOCAL_SPEC
    old.assert_hashes(frozen['hashes'],local.P)
    old.assert_hashes(read(local.P/'INPUT_HASHES.json'));old.assert_hashes(read(local.P/'CACHE_HASHES.json'),local.P)
    validate_local_completion(read(local.P/'VALIDATION.json'))
    assert scope.configs()[3]==SOURCE_SCOPE_SPEC


def validate_local_completion(value):
    for key in ('original_controls_exact','independent_fixed_unit_replay','training_freeze_parity','source_model_parity'):
        assert value.get(key) is True,'Local control not verified: '+key
    assert value.get('protected_files')==21436 and value.get('protected_changes')==[]
    assert value.get('observer_accounts_changed') is False
    assert value.get('fixed_configs')==5 and value.get('slot_replacement_fraction')==.25


def assert_original_arrays(original,component):
    for key,value in original.items():
        assert key in component,'Missing inherited source array: '+key
        left,right=np.asarray(value),np.asarray(component[key])
        assert left.dtype==right.dtype,'Inherited dtype changed: '+key
        np.testing.assert_array_equal(left,right,err_msg='Inherited source changed: '+key)


def validate_calendar(current,reference):
    """Same official clock; each component pins its complete import graph."""
    assert current['provenance']==reference['provenance'],'Official calendar provenance differs'
    common=set(current['source_hashes'])&set(reference['source_hashes'])
    assert common,'Calendar dependency graphs have no shared source'
    for path in common:
        assert current['source_hashes'][path]==reference['source_hashes'][path],'Shared calendar source changed: '+path


def source_bundle():
    m,a,px,op=scope.load();lm,la,lp,lo=local.load()
    assert lm==m,'Local control metadata differs'
    assert_original_arrays(a,la)
    pd.testing.assert_frame_equal(lp,px,check_exact=True);pd.testing.assert_frame_equal(lo,op,check_exact=True)
    policy=read(scope.P/'INCUMBENT_POLICY.json')
    assert read(local.P/'INCUMBENT_POLICY.json')==policy,'Local control policy differs'
    protected=read(scope.P/'PROTECTED.json');assert len(protected)==21436
    assert read(local.P/'PROTECTED.json')==protected
    calendar=calendar_provenance()
    reference=local.calendar_provenance()
    assert reference==read(local.P/'PLAN.json')['calendar']
    validate_calendar(calendar,reference)
    return m,la,px,op,policy


def raw_source_hashes(info):
    hashes=dict(info['source_hashes']);loader=O/'monthly_stability_monitor/inputs.py'
    assert hashes.get(str(loader))==sha(loader) and len(hashes)==64
    source=read(target.P/'INPUT_HASHES.json')
    assert all(source.get(path)==digest for path,digest in hashes.items()),'Original raw input changed'
    return hashes


def initialize():
    assert not (P/'PLAN.json').exists(),'Never overwrite registration'
    fixed=design();verify_source();source_bundle()
    _,info=arch.old.panel_load(read(scope.P/'META.json')['etfs'])
    protected=read(scope.P/'PROTECTED.json');assert len(protected)==21436
    dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        design_sha256=DESIGN_SHA256,design=fixed,
        raw_source_hashes=raw_source_hashes(info),calendar=calendar_provenance(),
        parity='Independently call original four-target model; source_log raw atol1e-12 rtol0, exact availability/ranks/four controls.',
        paired='Exact original monthly anchors vs their union with official W-FRI anchors; same log21 labels on original A rows.',
        weight='Both annual weight sums equal mature monthly row count N_A; B restricted to A mature calendar months.',
        candidate_start='Original scope67 pre; local-linear quarter is control3/default benchmark only.',
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
    if spec not in configs():raise ValueError('Only six fixed configurations permitted')
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


def ranked_forecasts(raw,available,source_raw,a,n):
    """Both arms use full original U, or jointly keep the complete source row."""
    shape=_shape(a,n)
    if set(raw)!=set(ARMS):raise ValueError('Exactly two matched candidate matrices required')
    values={key:adapter._matrix(raw[key],shape,n,key) for key in ARMS}
    if any(np.isinf(value).any() for value in values.values()):raise FloatingPointError('Infinite density forecast')
    available=adapter._matrix(available,shape,n,'candidate availability',boolean=True)
    source_raw=adapter._matrix(source_raw,shape,n,'independent source raw')
    original,use=original_rank_inputs(a,n)
    if not np.isfinite(source_raw[use]).all():raise ValueError('Source predictions must be finite inside U')
    ranks={key:original.copy() for key in ARMS};ready=np.zeros(n,bool)
    for i in range(n):
        ready[i]=bool(use[i].any() and available[i,use[i]].all()
            and all(np.isfinite(value[i,use[i]]).all() for value in values.values()))
        if ready[i]:
            for key in ARMS:
                if np.array_equal(values[key][i,use[i]],source_raw[i,use[i]]):continue
                rank=ep.old.prior.ranks(pd.Series(values[key][i]).where(use[i]))
                ranks[key][i,use[i]]=rank[use[i]]
    return ranks,ready


def score_delta(spec,a,n):
    source_spec(spec);shape=_shape(a,n);delta=np.zeros((n,shape[1]))
    if spec['mode']=='none':return delta
    ranks,ready=ranked_forecasts({key:a['density_raw_'+key] for key in ARMS},a['density_available'],a['density_source_raw'],a,n)
    current,use=original_rank_inputs(a,n)
    weak=adapter._state(a,'state_weak_narrow',shape[0],n)
    calm=adapter._state(a,'state_calm10',shape[0],n)
    extension=adapter._matrix(a['extension_available'],shape,n,'extension availability',boolean=True)
    coefficient=.05*weak[:,None]+.01*(calm&~weak)[:,None]*extension
    active=use&ready[:,None]
    delta[active]=coefficient[active]*(ranks[spec['mode']][active]-current[active])
    return delta


def modified_pre_score(spec,a,n):
    source=source_spec(spec);_shape(a,n)
    if spec['base']=='incumbent_local67':return local.modified_pre_score(LOCAL_SPEC,a,n)
    pre=scope.modified_pre_score(source,a,n)
    delta=score_delta(spec,a,n);changed=delta!=0.
    view=pre[:n].copy();view[changed]+=delta[changed];pre[:n]=view
    return pre


def choices(spec,a,m,n,*,policy=None,boundary_selector=None):
    source=source_spec(spec);shape=_shape(a,n)
    if spec['base']!='original65':
        policy=read(P/'INCUMBENT_POLICY.json') if policy is None else policy
        assert all(policy.get(k)==v for k,v in POLICY_FIELDS.items())
    if spec['base']=='incumbent_local67':
        return local.choices(LOCAL_SPEC,a,m,n,policy=policy,boundary_selector=boundary_selector)
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


def validate_feature_inputs(data,cached,train_meta):
    arrays,meta=data['arrays'],data['meta']
    if set(arrays)!=set(('density_X','density_eligible','density_dates','density_symbols','density_is_monthly','monthly_row_index')):
        raise ValueError('Wrong density feature keys')
    X=np.asarray(arrays['density_X']);rows=len(X);g=np.asarray(arrays['density_eligible']);monthly=np.asarray(arrays['density_is_monthly'])
    idx=np.asarray(arrays['monthly_row_index']);days=np.asarray(arrays['density_dates']);symbols=np.asarray(arrays['density_symbols'])
    if X.shape!=(rows,10) or X.dtype!=np.asarray(cached['train_plain']).dtype:raise ValueError('Original floating plain10 dtype/width required')
    if g.shape!=(rows,) or monthly.shape!=(rows,) or g.dtype.kind!='b' or monthly.dtype.kind!='b':raise ValueError('Boolean eligible/monthly axes required')
    if days.shape!=(rows,) or days.dtype.kind!='M' or np.isnat(days).any():raise ValueError('Finite flat signal dates required')
    if symbols.shape!=(rows,) or symbols.dtype.kind not in 'US':raise ValueError('Explicit flat security IDs required')
    if idx.ndim!=1 or idx.dtype.kind not in 'iu':raise ValueError('Integer monthly mapping required')
    np.testing.assert_array_equal(idx,np.flatnonzero(monthly))
    widths=np.array([len(pool) for pool in meta['symbols']],int)
    assert len(meta['dates'])==len(meta['symbols'])==len(meta['pool_origins'])==len(meta['is_monthly'])
    assert all(len(pool)==len(set(pool)) and all(isinstance(z,str) and z for z in pool) for pool in meta['symbols'])
    np.testing.assert_array_equal(meta['offsets'],np.r_[0,np.cumsum(widths)])
    np.testing.assert_array_equal(days,np.repeat(np.array(meta['dates'],dtype='datetime64[D]'),widths))
    np.testing.assert_array_equal(symbols,np.array([z for pool in meta['symbols'] for z in pool]))
    np.testing.assert_array_equal(monthly,np.repeat(np.asarray(meta['is_monthly'],bool),widths))
    np.testing.assert_array_equal(days[idx],cached['dates'])
    np.testing.assert_array_equal(symbols[idx],np.array([z for pool in train_meta['symbols'] for z in pool]))
    np.testing.assert_array_equal(X[idx],cached['train_plain'])
    np.testing.assert_array_equal(g[idx],cached['train_eligible'])
    return dict(monthly_rows=len(idx),union_rows=rows,monthly_rows_mapped_exact=True,
        explicit_symbols_dates_offsets=True,original_eligibility_and_plain10_exact=True)


def validate_labels(arrays,rows,days):
    if set(arrays)!=set(('density_y','density_ends','density_good')):raise ValueError('Wrong density label keys')
    y=np.asarray(arrays['density_y']);end=np.asarray(arrays['density_ends']);good=np.asarray(arrays['density_good'])
    if y.shape!=(rows,) or y.dtype.kind!='f' or good.shape!=(rows,) or good.dtype.kind!='b':raise ValueError('Invalid label axes')
    if end.shape!=(rows,) or end.dtype.kind!='M' or np.isnat(end).any() or np.any(end<=days):raise ValueError('Label ends must be known future sessions')
    if np.isinf(y).any() or not np.array_equal(np.isfinite(y),good):raise FloatingPointError('Finite label mask must equal declared training good')


def validate_monthly_labels(union,monthly,idx):
    for key in ('density_y','density_ends','density_good'):
        assert union[key].dtype==monthly[key].dtype,'Monthly label dtype changed: '+key
        np.testing.assert_array_equal(union[key][idx],monthly[key],err_msg='Adding weekly anchors changed original monthly labels: '+key)
    return dict(independent_monthly_rebuild=True,all_monthly_labels_ends_and_good_exact=True,
        monthly_good_rows=int(monthly['density_good'].sum()),union_good_rows=int(union['density_good'].sum()))


def validate_forecasts(forecast,rows):
    if set(forecast['raw'])!=set(('source_log',*ARMS)):raise ValueError('Wrong forecast keys')
    for key in ('available','candidate_available'):
        value=np.asarray(forecast[key])
        if value.shape!=(rows,) or value.dtype.kind!='b':raise ValueError('Invalid '+key)
    if (forecast['candidate_available']&~forecast['available']).any():raise ValueError('Candidate availability must not expand source')
    for key,value in forecast['raw'].items():
        value=np.asarray(value);mask=forecast['available'] if key=='source_log' else forecast['candidate_available']
        if value.shape!=(rows,) or not np.array_equal(np.isfinite(value),mask) or np.isinf(value).any():
            raise FloatingPointError('Forecast must be finite exactly on declared availability: '+key)


def prepare():
    registered()
    assert not (P/'DATA.npz').exists() and not (P/'FROZEN.json').exists(),'Never overwrite prepared/frozen data'
    verify_source();m,a,px,op,policy=source_bundle()
    with np.load(target.P/'TRAIN_INPUTS.npz') as stored:cached={key:stored[key].copy() for key in stored.files}
    with np.load(target.old.P/'DATA.npz') as stored:original={key:stored[key].copy() for key in target.CACHED_KEYS}
    meta=read(target.old.P/'META.json')
    identity=validate_identity(meta,cached,original,read(arch.P/'META.json'))
    target.validate_axes(m,meta,cached,a['eligible'])
    assert cached['train_plain'].shape==(8820,10) and cached['event_plain'].shape==(5978,10)
    assert a['baseline'].shape==(122,49)
    panel,info=arch.old.panel_load(m['etfs']);raw=raw_source_hashes(info)
    assert raw==read(P/'PLAN.json')['raw_source_hashes']
    parity=verify_panel_parity(panel,px,op)
    legacy=pd.read_csv(features.LEGACY_PATH,index_col=0,parse_dates=True).fillna(False).astype(bool)
    inputs=features.build_inputs(panel,meta['training'],m,cached['train_plain'],cached['event_plain'],
        cached['train_eligible'],a['eligible'],legacy_member=legacy)
    input_check=validate_feature_inputs(inputs,cached,meta['training']);density=inputs['arrays']
    union_labels=labels.build_labels(panel['adj'],inputs['meta'],density['density_X'],density['density_eligible'])
    # Build A independently: never declare parity by slicing B into both sides.
    monthly_labels=labels.build_labels(panel['adj'],meta['training'],cached['train_plain'],cached['train_eligible'])
    validate_labels(union_labels['arrays'],len(density['density_dates']),density['density_dates'])
    validate_labels(monthly_labels['arrays'],len(cached['dates']),cached['dates'])
    label_check=validate_monthly_labels(union_labels['arrays'],monthly_labels['arrays'],density['monthly_row_index'])
    density.update(union_labels['arrays'])
    source_targets={name:cached['target_'+name] for name in target.TARGETS}
    event_good=a['eligible'].reshape(-1)&np.isfinite(cached['event_plain']).all(axis=1)
    forecast=model.annual_forecasts(cached['dates'],cached['ends'],cached['train_plain'],source_targets,
        cached['common_good'],cached['event_dates'],cached['event_plain'],event_good,
        density['density_dates'],density['density_ends'],density['density_X'],density['density_y'],
        density['density_good'],density['density_is_monthly'])
    validate_forecasts(forecast,len(cached['event_dates']))
    shape=a['baseline'].shape;source_available=forecast['available'].reshape(shape)
    np.testing.assert_array_equal(source_available,a['forecast_available'])
    source_raw=forecast['raw']['source_log'].reshape(shape)
    np.testing.assert_allclose(source_raw,a['raw_log_excess'],rtol=0.,atol=PARITY_ATOL,equal_nan=True)
    np.testing.assert_array_equal(source_ranked_forecast(source_raw,a),a['rank_log_excess'])
    a['density_source_raw']=source_raw;a['density_model_available']=source_available
    a['density_available']=forecast['candidate_available'].reshape(shape)
    for key in ARMS:a['density_raw_'+key]=forecast['raw'][key].reshape(shape)
    ranks,ready=ranked_forecasts({key:a['density_raw_'+key] for key in ARMS},a['density_available'],source_raw,a,len(m['dates']))
    for key in ARMS:a['density_rank_'+key]=ranks[key]
    a['density_month_ready']=ready
    controls={}
    for ident,sourceid in ((0,0),(1,1),(2,3)):
        got=choices(configs()[ident],a,m,len(m['dates']),policy=policy)
        expected=scope.choices(scope.configs()[sourceid],a,m,len(m['dates']),policy=policy)
        assert got[0]==expected[0];np.testing.assert_array_equal(got[1],expected[1])
        filename=('original65_ORDERS.json','incumbent66_ORDERS.json','primary_ORDERS.json')[ident]
        assert rp.make_orders(*got,m)==read(scope.P/filename)
        controls[ident]=got
    local_control=choices(configs()[3],a,m,len(m['dates']),policy=policy)
    expected=local.choices(LOCAL_SPEC,a,m,len(m['dates']),policy=policy)
    assert local_control[0]==expected[0];np.testing.assert_array_equal(local_control[1],expected[1])
    assert rp.make_orders(*local_control,m)==read(local.P/'local_linear_ORDERS.json')
    controls[3]=local_control
    fallback=dict(a,density_available=np.zeros(shape,bool))
    for key in ARMS:fallback['density_raw_'+key]=np.full(shape,np.nan)
    for spec in configs()[4:]:
        got=choices(spec,fallback,m,len(m['dates']),policy=policy)
        assert got[0]==controls[2][0];np.testing.assert_array_equal(got[1],controls[2][1])
        assert rp.make_orders(*got,m)==rp.make_orders(*controls[2],m)
        full=choices(spec,a,m,len(m['dates']),policy=policy);prefix=choices(spec,a,m,TRAIN,policy=policy)
        assert prefix[0]==full[0][:TRAIN];np.testing.assert_array_equal(prefix[1],full[1][:TRAIN])
    np.savez_compressed(P/'DENSITY_INPUTS.npz',**density)
    np.savez_compressed(P/'MONTHLY_LABELS.npz',**monthly_labels['arrays'])
    np.savez_compressed(P/'DATA.npz',**a)
    dump('META.json',m);dump('DENSITY_META.json',inputs['meta']);dump('INCUMBENT_POLICY.json',policy)
    dump('IDENTITY_AUDIT.json',identity);dump('FEATURE_AUDIT.json',dict(inputs['audit'],driver_validation=input_check))
    dump('LABEL_AUDIT.json',dict(union=union_labels['audit'],monthly=monthly_labels['audit'],parity=label_check))
    dump('MODEL_AUDIT.json',forecast['audit'])
    original_use=a['forecast_available']&a['eligible']
    dump('RANK_AUDIT.json',dict(population='original forecast_available & original eligible for all query months',
        original_population_cells=int(original_use.sum()),candidate_ready_months=int(ready.sum()),
        fallback_months=int((original_use.any(axis=1)&~ready).sum()),
        empty_original_population_months=int((~original_use.any(axis=1)).sum()),
        both_arms_share_wholemonth_fallback=True,no_training_cohort_query_gate=True,no_ranking_population_shrink=True,
        all_equal_source_reuses_cached_rank=True,whole_month_fallback_warning=bool((original_use.any(axis=1)&~ready).any())))
    dump('PREPARE_VALIDATION.json',dict(all_four_source_controls_scores_orders_exact=True,
        source_model_internal_only=True,source_log_raw_atol=PARITY_ATOL,source_availability_rank_exact=True,
        source_label_cache_exact=True,original_monthly_query_plain10_and_eligibility_exact=True,
        independent_monthly21_labels_exact=True,missing_exact_current_scope=True,all_candidates_prefix_exact=True,
        no_new_ml_budget=True,fixed_existing_slot_replacement_fraction=1.,original_support_payload_exact=True,raw_panel_parity=parity))
    hashes=target.merge_hashes(read(scope.P/'INPUT_HASHES.json'),read(target.P/'INPUT_HASHES.json'),read(local.P/'INPUT_HASHES.json'),
        info['source_hashes'],calendar_provenance()['source_hashes'])
    for f in list(implementations())+list(source_files())+[P/'DESIGN.json',P/'DESIGN.sha256',P/'DESIGN.initial.json',P/'DESIGN.initial.sha256']:
        digest=sha(f);assert str(f) not in hashes or hashes[str(f)]==digest,str(f);hashes[str(f)]=digest
    dump('INPUT_HASHES.json',hashes)
    dump('CACHE_HASHES.json',{f:sha(P/f) for f in ('DATA.npz','DENSITY_INPUTS.npz','MONTHLY_LABELS.npz','DENSITY_META.json','META.json','INCUMBENT_POLICY.json',
        'IDENTITY_AUDIT.json','FEATURE_AUDIT.json','LABEL_AUDIT.json','MODEL_AUDIT.json','RANK_AUDIT.json','PREPARE_VALIDATION.json','INPUT_HASHES.json')})
    print('Prepared six fixed matched anchor-density configurations and four exact controls',flush=True)


def load():
    registered();verify_source()
    old.assert_hashes(read(P/'INPUT_HASHES.json'));old.assert_hashes(read(P/'CACHE_HASHES.json'),P)
    m,_,px,op,_=source_bundle();assert m==read(P/'META.json')
    with np.load(P/'DATA.npz') as data:a={key:data[key].copy() for key in data.files}
    ranks,ready=ranked_forecasts({key:a['density_raw_'+key] for key in ARMS},a['density_available'],a['density_source_raw'],a,len(m['dates']))
    for key in ARMS:np.testing.assert_array_equal(ranks[key],a['density_rank_'+key])
    np.testing.assert_array_equal(ready,a['density_month_ready'])
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
    selected = dict(primary=primary,original65=rows[0],incumbent66=rows[1],incumbent_scope67=rows[2],incumbent_local67=incumbent)
    selected.update({diagnostic_name(z['spec']):z for z in rows[4:]})
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
        control.extend(returns[3][bounds[n]:bounds[end]])
        folds.append(dict(train_end=m['ends'][n-1],test_end=m['ends'][end-1],spec=winner['spec']))
    dump('WALKFORWARD.json',dict(folds=folds,pristine_oos=False,
        deployment='Stateless monthly selection; each fold uses only its selected fixed rule, with no prior-holding dependency.',
        metrics=rp.stats_rr(np.array(stitched),m['dates'][36],m['ends'][87]),
        incumbent=rp.stats_rr(np.array(control),m['dates'][36],m['ends'][87])))
    dump('TRAIN_TRIALS.json',rows)
    np.savez_compressed(P/'TRAIN_RETURNS.npz',returns=np.stack(returns),bounds=np.array(bounds))
    files=('DESIGN.json','DESIGN.sha256','PLAN.json','GRID.json','DATA.npz','META.json','INCUMBENT_POLICY.json',
           'IDENTITY_AUDIT.json','FEATURE_AUDIT.json','LABEL_AUDIT.json','MODEL_AUDIT.json','RANK_AUDIT.json',
           'DENSITY_INPUTS.npz','MONTHLY_LABELS.npz','DENSITY_META.json','PREPARE_VALIDATION.json','INPUT_HASHES.json',
           'CACHE_HASHES.json','PROTECTED.json','study.py','model.py','features.py','labels.py','test_study.py','test_model.py','test_features.py','test_labels.py',
           'DESIGN.initial.json','DESIGN.initial.sha256')
    dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),selected=selected,
        trials=6,new_admitted=sum(int(z['admitted']) for z in rows[4:]),
        primary_source=selected['primary']['spec'],
        source_log_internal_parity_only=True,matched_annual_month_support_and_weight_total=True,
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
    for name,reference in [(diagnostic_name(z),reference) for z in configs()[4:] for reference in ('incumbent_local67','incumbent_scope67')]+[('union21','month21')]:
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
        observer_accounts_changed=False,pristine_oos=False,fixed_configs=6,
        core_coefficients_unchanged=True,existing_weak_slot=.05,existing_calm_nonweak_slot=.01,nominal_episode_budget=.05,both_diagnostics_frozen=True,
        modified_pre_drives_top2=True,original_support_payload_exact=True,true_episode_boundary_rules=True,
        no_global_rerank=True,no_intramonth_transactions=True,source_plain10_exact=True,annual_clock_unchanged=True,matched_log21_targets=True,
        source_model_parity=True,matched_annual_month_support_and_weight_total=True,
        candidate_vs_source_attribution=True,matched_anchor_attribution=True,slot_replacement_fraction=1.,
        primary_target70_reached=results['primary']['metrics']['cagr']>=.70))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('initialize','prepare','train','evaluate'))
    globals()[parser.parse_args().phase]()
