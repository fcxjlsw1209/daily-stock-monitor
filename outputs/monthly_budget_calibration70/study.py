"""Fixed guidance budgets versus annual constrained four-factor calibration.

Feature preparation never fits the new model or chooses either new portfolio.
Independent feature and model reviews gate the two later stages separately.
"""
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


local = module('budget_calibration70_local', O/'monthly_local_linear70/study.py')
scope, target = local.scope, local.target
old, rp, ex, arch, ep, read, sha, adapter = (local.old, local.rp, local.ex,
    local.arch, local.ep, local.read, local.sha, local.adapter)
features = module('budget_calibration70_features', P/'features.py')
model = module('budget_calibration70_model', P/'model.py')
ordinal = module('budget_calibration70_ordinal', O/'monthly_core_expansion/ordinal_model.py')
TRAIN = 88
LOCAL_SPEC = dict(id=4, base='incumbent_scope67', mode='local_linear')
BOUNDARY = ROOT/'work/monitor_research_boundary/settlement_20261014T200412Z'
PROTECTED_COUNT = 21444
DESIGN_SHA256 = '995fb42776fb0b0dece659997b2b4f0ecd642066093f3e95a95b405cb51bcf35'
ACTUAL_BOUND_FILES = ('DATA.npz', 'META.json', 'FEATURE_AUDIT.json',
                      'INPUT_HASHES.json', 'CACHE_HASHES.json', 'PLAN.json')
POLICY_FIELDS = dict(local.POLICY_FIELDS)
MODEL_BOUND_FILES = ('DATA.npz','META.json','BUDGET_WEIGHTS.npz','MODEL_AUDIT.json',
                     'PLAN.json','CACHE_HASHES.json','INPUT_HASHES.json')
FEATURE_KEYS = tuple('budget_'+key for key in ('train_X','event_X','train_dates','train_ends',
    'train_symbols','train_offsets','train_eligible','event_eligible','train_target',
    'train_label_valid','event_dates'))
ARMS = ('guidance4_fixed','annualcalibrated4f')
PRIOR=np.array([.65,.15,.10,.10])
WEIGHT_TOL=1e-9
PRICE_SOURCE_FILES = (O/'monthly_stability_monitor/inputs.py',O/'index_union/price_units.py')
MAPPING_SOURCE = Path(ex.t.strict.__file__)
SECTOR_SOURCE = O/'index_union/alias_resolved_inputs/sectors.csv.gz'
TRAIN_META=O/'monthly_blend_optimization/MONTHLY_TRAIN.json'
TRAIN_CACHE=TRAIN_META.with_suffix('.npz')


def dump(name, value):
    (P/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else x)+'\n')


def configs():
    return [dict(id=0, base='original65', mode='none'),
            dict(id=1, base='incumbent66', mode='none'),
            dict(id=2, base='incumbent_scope67', mode='none'),
            dict(id=3, base='incumbent_local67', mode='none'),
            dict(id=4, base='guidance4', mode='fixed'),
            dict(id=5, base='guidance4', mode='annual_calibrated')]


def diagnostic_name(spec):
    if spec not in configs()[4:]:
        raise ValueError('Only preregistered guidance candidates allowed')
    return ARMS[spec['id']-4]


def implementations():
    return tuple(sorted(set(local.implementations()) | set(model.dependency_files()) |
        {P/name for name in ('study.py','test_study.py','report.py','features.py','test_features.py','model.py','test_model.py')} |
        set(PRICE_SOURCE_FILES) | {MAPPING_SOURCE} |
        {Path(adapter.__file__),Path(arch.__file__),Path(arch.old.__file__),
         Path(ordinal.__file__),ordinal._FILE}))


def source_files():
    files = set(local.source_files())
    files.update(local.P/name for name in ('DESIGN.json', 'DESIGN.sha256', 'PLAN.json',
        'GRID.json', 'FROZEN.json', 'FROZEN.sha256', 'DATA.npz', 'META.json', 'INPUT_HASHES.json',
        'CACHE_HASHES.json', 'INCUMBENT_POLICY.json', 'VALIDATION.json', 'RESULTS.json',
        'original65_ORDERS.json', 'incumbent66_ORDERS.json', 'incumbent_scope67_ORDERS.json',
        'local_linear_ORDERS.json'))
    files.update(BOUNDARY/name for name in ('BEFORE.json', 'AFTER.json',
        'RESEARCH_BOUNDARY.json', 'NEXT_EXPECTED_PROTECTED.json'))
    files.update(PRICE_SOURCE_FILES)
    files.update((MAPPING_SOURCE,SECTOR_SOURCE,TRAIN_META,TRAIN_CACHE))
    files.update(ROOT/path for path in read(P/'DESIGN.json')['scope_reviews'])
    return tuple(sorted(files))


def design():
    digest = (P/'DESIGN.sha256').read_text().strip()
    assert sha(P/'DESIGN.json') == digest == DESIGN_SHA256
    value = read(P/'DESIGN.json')
    assert value['configs'] == configs()
    assert value['boundary'] == str(BOUNDARY) and value['protected_count'] == PROTECTED_COUNT
    assert value['feature_order']==['M','R','V','P'] and value['fixed_prior']==PRIOR.tolist()
    for path,expected in value['scope_reviews'].items():assert sha(ROOT/path)==expected,path
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
    assert len(raw)==64
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
        raise ValueError('Only six fixed configurations permitted')
    return local.configs()[spec['id']] if spec['id'] < 3 else LOCAL_SPEC


def weight_rows(values,dates,n):
    """Validate only the requested event prefix; never normalize fitted weights."""
    value=np.asarray(values)
    if value.shape!=(len(dates),4):raise ValueError('Weights must have shape[events,4]')
    w=value[:n].astype(float,copy=False)
    days=np.asarray(dates[:n],dtype='datetime64[D]')
    if np.isnat(days).any() or not np.isfinite(w).all():raise ValueError('Unknown date or nonfinite weights')
    if ((w < -WEIGHT_TOL).any() or (w[:,0]<.60-WEIGHT_TOL).any()
            or (np.abs(w.sum(axis=1)-1.)>WEIGHT_TOL).any()
            or (np.abs(w-PRIOR).sum(axis=1)>.10+WEIGHT_TOL).any()):
        raise ValueError('Weights violate fixed simplex/core/L1 constraints')
    years=days.astype('datetime64[Y]')
    for year in np.unique(years):
        rows=w[years==year]
        if not np.array_equal(rows,np.broadcast_to(rows[0],rows.shape)):
            raise ValueError('Every event in a year must use identical weights')
    return w


def score(spec,a,m,n):
    diagnostic_name(spec)
    shape=local._shape(a,n)
    values=np.asarray(a['budget_event_X'])
    if values.shape!=(*shape,4):raise ValueError('Expected four event factors on original axes')
    x=values[:n].astype(float,copy=False)
    if not np.isfinite(x).all() or (np.abs(x)>.5).any():
        raise ValueError('Prepared ranks must be finite in [-.5,.5]; no trade-time filling')
    dates=m['dates']
    if len(dates)!=shape[0]:raise ValueError('Event dates must align with original month axis')
    w=np.broadcast_to(PRIOR,(n,4)) if spec['mode']=='fixed' else weight_rows(a['budget_event_weights'],dates,n)
    # No BLAS layout dependence, post-fit rank or intercept. Exact w0 is common
    # to fixed and annual coldstart, preserving their score arithmetic order.
    out=np.einsum('ijk,ik->ij',x,w,optimize=False)
    if not np.isfinite(out).all() or (np.abs(out)>.5+WEIGHT_TOL).any():
        raise FloatingPointError('Invalid bounded four-factor score')
    return out


def choices(spec,a,m,n,*,policy=None,boundary_selector=None):
    source=source_spec(spec)
    if spec['mode']=='none':
        return local.choices(source,a,m,n,policy=policy,boundary_selector=boundary_selector)
    shape=local._shape(a,n)
    good=adapter._matrix(a['eligible'],shape,n,'eligibility',boolean=True)
    symbols=m['symbols'][:n]
    if len(symbols)!=n or any(len(ss)!=shape[1] or len(set(ss))!=len(ss)
            or any(not isinstance(t,str) or not t for t in ss) for ss in symbols):
        raise ValueError('Security identities must match the original event axes')
    sc=score(spec,a,m,n)
    # Both new rules are complete all-seat architectures; no old model,
    # episode, top2 lock, cap or extra filter is called here.
    return rp.choose(sc,good,symbols,3),sc


def verify_inherited_arrays(current,source):
    assert len(source)==27,'Unexpected inherited source array set'
    for key,expected in source.items():
        actual=current[key]
        assert actual.shape==expected.shape and actual.dtype==expected.dtype,key
        assert actual.tobytes()==expected.tobytes(),'Inherited array drift: '+key


def validate_features(arrays,m,tm,source_eligible=None):
    if set(arrays)!=set(FEATURE_KEYS):raise ValueError('Expected exactly11 prefixed feature/label arrays')
    widths=np.array([len(ss) for ss in tm['symbols']],dtype=np.int64)
    offsets=np.r_[0,np.cumsum(widths)]
    if len(tm['dates'])!=len(widths) or len(tm['ends'])!=len(widths):raise ValueError('Training axes differ')
    n=int(offsets[-1]);q=len(m['dates']);width=len(m['symbols'][0])
    if any(len(ss)!=width or len(set(ss))!=width for ss in m['symbols']):raise ValueError('Event pool identity drift')
    def require(key,shape,kind):
        v=np.asarray(arrays['budget_'+key])
        if v.shape!=shape or v.dtype.kind not in kind:raise ValueError('Invalid '+key+' dtype/shape')
        return v
    train=require('train_X',(n,4),'f');event=require('event_X',(q,width,4),'f')
    for value in (train,event):
        if not np.isfinite(value).all() or (np.abs(value)>.5).any():raise ValueError('All four source ranks must be finite and bounded')
    oldoffset=require('train_offsets',(len(offsets),),'iu')
    np.testing.assert_array_equal(oldoffset,offsets)
    for key,dates in (('train_dates',np.repeat(np.array(tm['dates'],dtype='datetime64[D]'),widths)),
                      ('train_ends',np.repeat(np.array(tm['ends'],dtype='datetime64[D]'),widths)),
                      ('event_dates',np.array(m['dates'],dtype='datetime64[D]'))):
        value=require(key,dates.shape,'M')
        if np.isnat(value).any() or value.dtype!=np.dtype('datetime64[D]'):raise ValueError('Unknown or non-daily '+key)
        np.testing.assert_array_equal(value,dates)
    symbols=require('train_symbols',(n,),'US')
    np.testing.assert_array_equal(symbols,np.array([t for ss in tm['symbols'] for t in ss]))
    eligible=require('train_eligible',(n,),'b');event_eligible=require('event_eligible',(q,width),'b')
    valid=require('train_label_valid',(n,),'b');target_value=require('train_target',(n,),'f')
    if np.isinf(target_value).any() or not np.array_equal(np.isfinite(target_value),valid):
        raise ValueError('Ordinal target finite mask must equal label validity')
    if (valid&~eligible).any() or (np.abs(target_value[valid])>.5).any():raise ValueError('Invalid eligible ordinal target')
    for lo,hi in zip(offsets[:-1],offsets[1:]):
        e=eligible[lo:hi];v=valid[lo:hi]
        if v.any() and not np.array_equal(v,e):raise ValueError('Partial eligible month labels are forbidden')
        if v.any() and abs(float(target_value[lo:hi][v].mean()))>1e-14:
            raise ValueError('Ordinal midpoint target must be centered within each complete month')
    if source_eligible is not None:
        np.testing.assert_array_equal(event_eligible,source_eligible)
    return dict(training_rows=n,training_months=len(widths),event_months=q,event_width=width,
        train_eligible_rows=int(eligible.sum()),event_eligible_cells=int(event_eligible.sum()),
        train_label_valid_rows=int(valid.sum()),explicit_flattened_identity=True,
        whole_month_ordinal_labels=True,no_future_label_trading_filter=True)


def prepare():
    registered()
    assert not any((P/name).exists() for name in ('DATA.npz','BUDGET_WEIGHTS.npz','FROZEN.json'))
    m,a,px,op=local.load()
    assert len(a)==27 and a['baseline'].shape==(122,49)
    inherited={key:value.copy() for key,value in a.items()}
    tm=read(TRAIN_META)
    with np.load(TRAIN_CACHE,allow_pickle=False) as z:source_offsets=z['offsets'].copy()
    np.testing.assert_array_equal(source_offsets,np.r_[0,np.cumsum([len(ss) for ss in tm['symbols']])])
    policy=read(local.P/'INCUMBENT_POLICY.json')
    expected_raw=read(P/'PLAN.json')['raw_source_hashes']
    panel,info=arch.old.panel_load(m['etfs'])
    actual_raw=local.raw_source_hashes(info)
    assert len(expected_raw)==64 and actual_raw==expected_raw
    assert str(SECTOR_SOURCE) in actual_raw
    parity=local.verify_panel_parity(panel,px,op)
    new,audit=features.prepare(panel,m,tm,raw_frames=arch.raw_frames,
        snapshot=arch.snapshot,label_builder=ordinal.build_targets)
    contract=validate_features(new,m,tm,a['eligible'])
    # Compare only original past eligibility/identity, never cached good/Y.
    with np.load(target.P/'TRAIN_INPUTS.npz',allow_pickle=False) as z:
        for source,newkey in (('offsets','train_offsets'),('dates','train_dates'),('ends','train_ends'),
                              ('train_eligible','train_eligible')):
            expected=z[source].astype('datetime64[D]') if source in ('dates','ends') else z[source]
            np.testing.assert_array_equal(new['budget_'+newkey],expected)
    for key,value in new.items():
        assert key not in a
        a[key]=value.copy()
    for spec in configs()[:4]:
        got=choices(spec,a,m,len(m['dates']),policy=policy)
        want=local.choices(source_spec(spec),a,m,len(m['dates']),policy=policy)
        assert got[0]==want[0] and got[1].tobytes()==want[1].tobytes()
        filename=('original65','incumbent66','incumbent_scope67','local_linear')[spec['id']]
        assert rp.make_orders(*got,m)==read(local.P/(filename+'_ORDERS.json'))
        for n in (0,1,36,60,76,TRAIN,len(m['dates'])):
            prefix=choices(spec,a,m,n,policy=policy)
            assert prefix[0]==got[0][:n]
            np.testing.assert_array_equal(prefix[1],got[1][:n])
    verify_inherited_arrays(a,inherited)
    assert set(a)==set(inherited)|set(FEATURE_KEYS) and len(a)==38
    audit=dict(audit,driver_validation=dict(contract,raw_source_hashes=actual_raw,raw_source_count=64,
        inherited_array_count=27,prepared_array_count=38,inherited_arrays_byte_exact=True,
        meta_byte_exact=True,original_training_offsets_and_eligibility_exact=True,
        mapping_source=dict(path=str(MAPPING_SOURCE),sha256=sha(MAPPING_SOURCE)),
        sector_source=dict(path=str(SECTOR_SOURCE),sha256=actual_raw[str(SECTOR_SOURCE)]),
        four_original_control_orders_exact=True,new_model_fit=False,new_candidate_choices=False))
    np.savez_compressed(P/'DATA.npz',**a)
    (P/'META.json').write_bytes((local.P/'META.json').read_bytes())
    assert sha(P/'META.json')==sha(local.P/'META.json')
    dump('INCUMBENT_POLICY.json',policy);dump('FEATURE_AUDIT.json',audit);dump('PRICE_PARITY.json',parity)
    dump('PREPARE_VALIDATION.json',dict(four_controls_exact=True,control_prefix_exact=True,
        source_raw_inputs=64,inherited_array_count=27,prepared_array_count=38,
        inherited_arrays_byte_exact=True,meta_byte_exact=True,
        original_event_and_training_eligibility_exact=True,new_model_fit=False,
        new_candidate_choices=False,no_trade_time_reranking=True,no_price_download=True,
        adjusted_price_and_opening_scale_parity=True,execution_prices_from_exact_local_cache=True))
    hashes=dict(read(local.P/'INPUT_HASHES.json'))
    for key,value in actual_raw.items():assert hashes.get(key)==value
    for f in list(implementations())+list(source_files())+[P/'DESIGN.json',P/'DESIGN.sha256']:
        value=sha(f)
        assert str(f) not in hashes or hashes[str(f)]==value
        hashes[str(f)]=value
    dump('INPUT_HASHES.json',hashes)
    dump('CACHE_HASHES.json',{f:sha(P/f) for f in ('DATA.npz','META.json','INCUMBENT_POLICY.json',
        'FEATURE_AUDIT.json','PRICE_PARITY.json','PREPARE_VALIDATION.json','INPUT_HASHES.json')})
    print('Prepared38 arrays and four controls; no new model fit or candidate selections',flush=True)


def load(*,with_weights=False):
    registered();verify_source()
    old.assert_hashes(read(P/'INPUT_HASHES.json'));old.assert_hashes(read(P/'CACHE_HASHES.json'),P)
    m,source,px,op=local.load()
    assert m==read(P/'META.json') and sha(P/'META.json')==sha(local.P/'META.json')
    with np.load(P/'DATA.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    verify_inherited_arrays(a,source)
    assert set(a)==set(source)|set(FEATURE_KEYS) and len(a)==38
    validate_features({key:a[key] for key in FEATURE_KEYS},m,read(TRAIN_META),source['eligible'])
    if with_weights:
        old.assert_hashes(read(P/'FIT_HASHES.json'),P)
        with np.load(P/'BUDGET_WEIGHTS.npz',allow_pickle=False) as z:
            assert set(z.files)=={'event_weights'}
            weights=z['event_weights'].copy()
        weight_rows(weights,m['dates'],len(m['dates']))
        a['budget_event_weights']=weights
    return m,a,px,op


def fit():
    registered();verify_actual_review()
    assert not any((P/name).exists() for name in ('BUDGET_WEIGHTS.npz','MODEL_AUDIT.json','FIT_HASHES.json','FROZEN.json'))
    m,a,_,_=load()
    weights,audit=model.annual_weights(a['budget_train_X'],a['budget_train_target'],
        a['budget_train_dates'],a['budget_train_ends'],a['budget_train_eligible'],
        a['budget_train_label_valid'],a['budget_event_dates'])
    weight_rows(weights,m['dates'],len(m['dates']))
    audit=dict(audit,driver_validation=dict(event_weights_shape=list(weights.shape),
        fixed_prior=PRIOR.tolist(),simplex_and_L1_constraints_checked=True,
        same_year_weights_identical=True,new_candidate_choices=False,
        feature_input_bindings={name:sha(P/name) for name in ACTUAL_BOUND_FILES}))
    np.savez_compressed(P/'BUDGET_WEIGHTS.npz',event_weights=weights)
    dump('MODEL_AUDIT.json',audit)
    dump('FIT_HASHES.json',{name:sha(P/name) for name in ('BUDGET_WEIGHTS.npz','MODEL_AUDIT.json')})
    print('Fitted annual budgets only; independent MODEL_REVIEW required before training',flush=True)


def verify_model_review():
    audit=read(P/'MODEL_REVIEW.json')
    assert audit.get('passed') is True,'Independent model audit has not passed'
    hashes=audit.get('hashes')
    assert isinstance(hashes,dict),'Independent model audit must bind its inputs'
    for name in MODEL_BOUND_FILES:
        assert hashes.get(name)==sha(P/name),'Independent model audit input drift: '+name
    return audit


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
    verify_model_review()
    assert not (P/'FROZEN.json').exists(), 'Never overwrite frozen selection'
    m,a,px,_ = load(with_weights=True)
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
           'FEATURE_AUDIT.json','PRICE_PARITY.json','PREPARE_VALIDATION.json','ACTUAL_FEATURE_REVIEW.json',
           'INPUT_HASHES.json','CACHE_HASHES.json','PROTECTED.json','study.py','test_study.py','report.py',
           'features.py','test_features.py','model.py','test_model.py','BUDGET_WEIGHTS.npz',
           'MODEL_AUDIT.json','MODEL_REVIEW.json','FIT_HASHES.json',
           'TESTS.txt','WALKFORWARD.json','TRAIN_TRIALS.json','TRAIN_RETURNS.npz')
    frozen_hashes={f:sha(P/f) for f in files}
    # Bind every true implementation dependency as well as the prepared cache.
    for path in set(implementations()) | set(source_files()):
        frozen_hashes[str(path)]=sha(path)
    dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),selected=selected,
        trials=6,new_admitted=sum(int(z['admitted']) for z in rows[4:]),
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
    m,a,px,op=load(with_weights=True)
    blocks,_,bounds,valid=rp.paths(px,m,len(m['dates']))
    orders={}
    for name,z in frozen['selected'].items():
        ch,score=choices(z['spec'],a,m,len(m['dates']))
        for n in (0,1,36,60,76,TRAIN,len(m['dates'])):
            prefix=choices(z['spec'],a,m,n)
            assert prefix[0]==ch[:n]
            np.testing.assert_array_equal(prefix[1],score[:n])
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
    comparisons=[(diagnostic_name(z),reference) for z in configs()[4:]
                 for reference in ('incumbent_local67','incumbent_scope67')]
    comparisons.append(('annualcalibrated4f','guidance4_fixed'))
    for name,reference in comparisons:
        changes=[]
        for row in months:
            z,b=row[name],row[reference]
            if set(z['symbols'])!=set(b['symbols']):
                changes.append(dict(date=row['date'],split=row['split'],old=b['symbols'],new=z['symbols'],
                    entering=sorted(set(z['symbols'])-set(b['symbols'])),
                    leaving=sorted(set(b['symbols'])-set(z['symbols'])),
                    difference=z['monthly_return']-b['monthly_return'],
                    log_contribution=float(np.log1p(z['monthly_return'])-np.log1p(b['monthly_return']))))
        positive=sorted([z['log_contribution'] for z in changes if z['log_contribution']>0],reverse=True)
        met=results[name]['metrics']
        undo=lambda n:float(((1+met['total_return'])*np.exp(-sum(positive[:n])))**(1/met['years'])-1)
        attribution[name+'__vs__'+reference]=dict(changed_months=len(changes),changes=changes,
            entering_seats=sum(len(z['entering']) for z in changes),
            leaving_seats=sum(len(z['leaving']) for z in changes),
            undo_best1_diagnostic_cagr=undo(1),undo_best3_diagnostic_cagr=undo(3))
    dump('ATTRIBUTION.json',attribution)
    years=np.array(m['dates'],dtype='datetime64[Y]')
    weight_summary=[]
    for year in np.unique(years):
        w=a['budget_event_weights'][np.flatnonzero(years==year)[0]]
        distance=float(np.abs(w-PRIOR).sum())
        weight_summary.append(dict(year=int(str(year)),weights=w.tolist(),
            L1_distance_from_prior=distance,core_lower_bound_hit=bool(abs(w[0]-.6)<=WEIGHT_TOL),
            L1_bound_hit=bool(abs(distance-.1)<=WEIGHT_TOL),
            nonnegative_bounds_hit=[int(j) for j in np.flatnonzero(np.abs(w)<=WEIGHT_TOL)],
            exact_prior=bool(np.array_equal(w,PRIOR))))
    dump('WEIGHT_SUMMARY.json',dict(feature_order=['M','R','V','P'],fixed_prior=PRIOR.tolist(),
        annual=weight_summary,weight_constraint_tolerance=WEIGHT_TOL,
        maximum_score_difference_from_fixed_bound=.05,all_three_seats_may_change=True))
    assert_protected()
    dump('RESULTS.json',results)
    dump('VALIDATION.json',dict(original_controls_exact=True,independent_fixed_unit_replay=True,
        training_freeze_parity=True,protected_files=len(read(P/'PROTECTED.json')),protected_changes=[],
        observer_accounts_changed=False,pristine_oos=False,fixed_configs=6,
        four_original_control_paths_unchanged=True,original_eligibility_unchanged=True,
        original_forecasts_unchanged=True,all_diagnostics_frozen=True,
        candidate_global_all_seat_selection=True,candidate_episode_calls=0,
        no_old_top2_or_boundary_constraint=True,four_factor_convex_scores=True,
        momentum_weight_at_least=.60,L1_prior_distance_at_most=.10,
        target_actual_monthly_ordinal_half=True,author_labels_used=False,
        no_intramonth_transactions=True,annual_clock_unchanged=True,
        raw_source_and_execution_cache_hashes_verified=True,
        prepared_ranks_not_reranked_at_selection=True,
        fixed_prior_matched_control=True,matched_pair_attribution=True,
        full_membership_feature_ranking=True,missing_source_rank_neutral_zero=True,
        inherited_array_count=27,prepared_array_count=38,meta_byte_exact=True,
        independent_actual_feature_audit_required_before_train=True,
        independent_model_review_required_before_train=True,
        prepare_does_not_fit_or_choose_new_rules=True,prefix_exact=True,
        primary_target70_reached=results['primary']['metrics']['cagr']>=.70))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('initialize','prepare','fit','train','evaluate'))
    globals()[parser.parse_args().phase]()
