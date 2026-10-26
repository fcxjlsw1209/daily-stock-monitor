"""Isolated fixed-incumbent reconstruction for one declared PTR pool sensitivity.

No search, source-cache mutation, or live-observer access. Control parity is
required before the conditional arm can be constructed or valued.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'outputs/monthly_ptr_pool_source169'
OUT=SRC/'fixed_incumbent_rebuild'
O=ROOT/'outputs'

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def stamp():return datetime.now(timezone.utc).isoformat()
def dump(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False,
                  default=lambda x:x.item() if isinstance(x,np.generic) else x)
        f.write('\n')

def libraries():
    local=module('ptr169_fixed_local',O/'monthly_local_linear70/study.py')
    identity=module('ptr169_identity_helpers',ROOT/'work/shell_pool_sensitivity168_identity.py')
    ordinal=module('ptr169_fixed_ordinal',O/'monthly_core_expansion/ordinal_model.py')
    return local,identity,ordinal

def register():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'PLAN.json').exists()
    assert (SRC/'conditional_pool/INDEPENDENT_REVIEW.json').exists(),'Independent pool review required'
    assert read(SRC/'conditional_pool/INDEPENDENT_REVIEW.json')['verdict']=='PASS_CONDITIONAL_DATA_SENSITIVITY_NOT_PIT_CERTIFICATION'
    local,identity,ordinal=libraries()
    paths=set(local.implementations())
    base=module('ptr169_base_dependencies',ROOT/'work/ptr169_rebuild_base.py')
    paths.update(base.dependency_files())
    paths.update(base.required_source_files())
    paths.update(Path(p) for p in read(local.P/'INPUT_HASHES.json'))
    paths.update([Path(__file__),ROOT/'work/test_ptr169_rebuild.py',ROOT/'work/ptr169_rebuild_base.py',Path(identity.__file__),
        Path(ordinal.__file__),local.P/'DATA.npz',local.P/'META.json',local.P/'INCUMBENT_POLICY.json',
        local.P/'local_linear_ORDERS.json',local.P/'INPUT_HASHES.json',
        O/'monthly_architecture_restart/META.json',O/'monthly_state_rank_research/TRAIN_DATA.json',
        O/'monthly_state_rank_research/TRAIN_DATA.npz',O/'monthly_state_rank_research/FEATURES.npz',
        O/'monthly_state_rank_research/FORECASTS.npz',O/'monthly_state_rank_research/FORECAST_AUDIT.json',
        O/'monthly_blend_optimization/MONTHLY_TRAIN.json',
        O/'monthly_2016_extension/POOLS.json',O/'monthly_2016_extension/PRICES.npz',
        O/'monthly_reconstruction_v2/foreign/membership.csv.gz',
        O/'monthly_top35_extension/POOLS.json',
        SRC/'conditional_pool/PLAN.json',SRC/'conditional_pool/DATA_COMPARISON.json',
        SRC/'conditional_pool/CONDITIONAL_POOLS.json',SRC/'conditional_pool/INDEPENDENT_REVIEW.json'])
    paths.update(ROOT/p for p in read(SRC/'conditional_pool/PLAN.json')['input_sha256'])
    for p in paths:assert p.is_file() and p.is_relative_to(ROOT),str(p)
    dump(OUT/'PLAN.json',dict(schema='ptr169_fixed_incumbent_rebuild_v1',registered_utc=stamp(),
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)},
        runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        fixed_candidate=local.configs()[4],episode_policy=read(local.P/'INCUMBENT_POLICY.json'),
        reference_orders='outputs/monthly_local_linear70/local_linear_ORDERS.json',
        preflight_notes=['Initial registration attempt stopped before creating PLAN because the proposed primary_ORDERS.json name did not exist. The explicit reference is the frozen local_linear_ORDERS.json for fixed config4; no reference or output was substituted during a model run.'],
        arms=['original_control','conditional_AH_fixed2015_laggedFX'],
        scope='Rebuild current incumbent only. No new strategy specification, coefficient search, boundary adjustment, or performance-based choice of pool.',
        periods=dict(start='2016-08-17',end='2026-09-24',events=122,early=53,training=88,later=34),
        method='Rebuild event identities and monthly/weekly training identities from original price panel. Recompute pool ranks, states, full-path label masks, four target Ridge, local residual Ridge and episode2. Preserve all original constants, annual cutoff conventions and boundary algorithm.',
        control_gate='Original 122 event and 180 monthly/783 weekly identities exact; eligible/missing/availability masks exact; numeric arrays atol1e-12 rtol0 where same units; fixed candidate scores within declared numeric tolerance, ordered holdings and 122 monthly order records exact. No conditional returns before parity.',
        failure='Write failure details, do not borrow cached forecasts to make a changed pool pass. Any implementation repair requires a new immutable revision and preserves this plan/code version.',
        data_limits=read(SRC/'conditional_pool/PLAN.json')['assumptions'],
        evaluation='If control passes, same-close zero, next-open zero, next-open 5bp per side; full/early53/recent35/later34/first88 using unchanged fixed-unit replay. Independent accounting and no promotion based on favorable conditional outcomes.',
        original_files_modified=False,observers_modified=False,new_strategy_search_trials=0))
    print(json.dumps(dict(status='REGISTERED',sha256=sha(OUT/'PLAN.json'))),flush=True)

def verify():
    plan=read(OUT/'PLAN.json')
    assert plan['runtime']==dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)
    for p,h in plan['input_sha256'].items():assert sha(ROOT/p)==h,p
    return plan

def context(conditional=False):
    local,identity,ordinal=libraries()
    reference_m,reference_a,px,op=local.load()
    panel,info=local.arch.old.panel_load(reference_m['etfs'])
    parity=local.verify_panel_parity(panel,px,op)
    m=copy.deepcopy(reference_m)
    pools=read(SRC/'conditional_pool/CONDITIONAL_POOLS.json' if conditional else O/'monthly_2016_extension/POOLS.json')
    for i,day in enumerate(m['dates']):m['symbols'][i]=identity.normalized_pool(pools[day],m['etfs'])
    if not conditional:assert m==reference_m,'Original event metadata mismatch'
    members={**{day:ss for day,ss in pools.items() if day<'2020-12-31'},
             **read(O/'monthly_top35_extension/POOLS.json')['32']}
    rebuilt_member=identity.member_table(members,panel['adj'].columns)
    if not conditional:pd.testing.assert_frame_equal(rebuilt_member,panel['member'])
    panel=dict(panel,member=rebuilt_member)
    tm=copy.deepcopy(read(O/'monthly_architecture_restart/META.json')['training'])
    legacy=pd.read_csv(O/'monthly_reconstruction_v2/foreign/membership.csv.gz',index_col=0,parse_dates=True).fillna(False).astype(bool)
    rows=identity.training_rows(tm['dates'],tm['ends'],panel['member'],legacy,set(m['etfs']))
    if not conditional:assert rows==identity.cached_rows(tm),'Original monthly identity mismatch'
    tm.update(symbols=[r['symbols'] for r in rows],pool_origins=[r['pool_origin'] for r in rows])
    assert len(m['dates'])==122 and len(rows)==180 and all(len(ss)==49 for ss in m['symbols']+tm['symbols'])
    return local,ordinal,panel,info,m,tm,px,op,reference_m,reference_a,parity

def monthly_features(local,ordinal,panel,m,tm):
    arch=local.arch
    f,mom,good=arch.raw_frames(panel)
    def snapshots(meta):
        values=[];eligibility=[]
        for day,ss in zip(meta['dates'],meta['symbols']):
            row,g,aux=arch.snapshot(day,ss,f,mom,good,panel,set(m['etfs']),'all')
            values.append(np.column_stack([row[k] for k in ['raw','recent_risk','vol','sector']]+[aux[:,j] for j in range(6)]))
            eligibility.append(g)
        return np.vstack(values),np.concatenate(eligibility)
    train,train_eligible=snapshots(tm);event,event_eligible=snapshots(m)
    offsets=np.r_[0,np.cumsum([len(ss) for ss in tm['symbols']])]
    dates=np.repeat(np.array(tm['dates'],dtype='datetime64[D]'),np.diff(offsets))
    ends=np.repeat(np.array(tm['ends'],dtype='datetime64[D]'),np.diff(offsets))
    event_dates=np.repeat(np.array(m['dates'],dtype='datetime64[D]'),49)
    symbols=np.array([s for ss in tm['symbols'] for s in ss])
    targets,ordinal_audit=ordinal.build_targets(panel['adj'],dates,ends,symbols,train_eligible)
    train_good=train_eligible&np.isfinite(targets['return'])&np.isfinite(targets['ordinal'])
    labels=local.target.labels.build_targets(panel['adj'],dates,ends,symbols,train_eligible,train_good,np.isfinite(train).all(axis=1))
    arrays=dict(train_plain=train,event_plain=event,train_eligible=train_eligible,train_good=train_good,
        dates=dates,ends=ends,event_dates=event_dates,offsets=offsets,
        eligible=event_eligible.reshape(122,49),common_good=labels['good'],vol63=labels['vol63'])
    arrays.update({'target_'+k:v for k,v in labels['targets'].items()})
    return arrays,dict(ordinal=ordinal_audit,targets=labels['audit'])

def models(local,a,cached,m,tm):
    targets={k:cached['target_'+k] for k in local.target.TARGETS}
    good=a['eligible'].reshape(-1)&np.isfinite(cached['event_plain']).all(axis=1)
    forecast=local.target.model.annual_forecasts(cached['dates'],cached['ends'],cached['train_plain'],
        targets,cached['common_good'],cached['event_dates'],cached['event_plain'],good)
    shape=a['eligible'].shape
    raw={k:forecast[k].reshape(shape) for k in local.target.TARGETS}
    available=forecast['available'].reshape(shape)
    ranks=local.target.ranked_forecasts(raw,available,a['eligible'],a['old_ml4'])
    a.update({'raw_'+k:v for k,v in raw.items()});a.update({'rank_'+k:v for k,v in ranks.items()})
    a['forecast_available']=available;a['extension_available']=available.copy()
    symbols=np.array([s for ss in tm['symbols'] for s in ss])
    event_symbols=np.array([s for ss in m['symbols'] for s in ss])
    result=local.model.annual_forecasts(cached['dates'],cached['ends'],symbols,cached['train_plain'],
        targets,cached['common_good'],cached['event_dates'],event_symbols,cached['event_plain'],good)
    local.validate_forecasts(result,len(event_symbols))
    np.testing.assert_array_equal(result['available'].reshape(shape),available)
    np.testing.assert_allclose(result['raw']['source_log'].reshape(shape),raw['log_excess'],atol=1e-12,rtol=0,equal_nan=True)
    a.update(raw_source_log=result['raw']['source_log'].reshape(shape),
             model_available=result['available'].reshape(shape),local_available=result['local_available'].reshape(shape))
    for k in local.ARMS:a['raw_'+k]=result['raw'][k].reshape(shape)
    ranked,ready=local.ranked_forecasts({k:a['raw_'+k] for k in local.ARMS},a['model_available'],a,len(m['dates']))
    for k in local.ARMS:a['rank_'+k]=ranked[k]
    a['paired_month_ready']=ready
    return a,dict(target=forecast['audit'],local=result['audit'])

def numeric_parity(actual,expected):
    assert set(actual)==set(expected),(sorted(set(actual)-set(expected)),sorted(set(expected)-set(actual)))
    result={}
    for k in expected:
        x,y=np.asarray(actual[k]),np.asarray(expected[k])
        assert x.shape==y.shape,(k,x.shape,y.shape)
        assert x.dtype.kind==y.dtype.kind,(k,'dtype kind',x.dtype,y.dtype)
        if x.dtype.kind in 'fc':
            np.testing.assert_array_equal(np.isfinite(x),np.isfinite(y),err_msg=k)
            np.testing.assert_allclose(x,y,atol=1e-12,rtol=0,equal_nan=True,err_msg=k)
            ok=np.isfinite(x)&np.isfinite(y)
            result[k]=dict(shape=list(x.shape),max_abs_error=float(np.max(np.abs(x[ok]-y[ok]))) if ok.any() else 0.)
        else:
            np.testing.assert_array_equal(x,y,err_msg=k);result[k]=dict(shape=list(x.shape),exact=True)
    return result

def validated_control():
    control=read(OUT/'control/PARITY.json')
    assert control['passed'] is True and control['arm']=='control'
    assert control['source_plan_sha256']==sha(OUT/'PLAN.json'),'Control uses different registered plan'
    required={'DATA.npz','META.json','ORDERS.json','TRAIN_INPUTS.npz','WEEKLY_INPUTS.npz',
              'MONTHLY_META.json','WEEKLY_META.json','BASE_DATA.npz','AUDIT.json',
              'BASE_AUDIT.json','LABEL_AUDIT.json','MODEL_AUDIT.json'}
    assert set(control['artifacts_sha256'])==required,'Incomplete control artifact attestation'
    for name,digest in control['artifacts_sha256'].items():
        assert sha(OUT/'control'/name)==digest,('Control artifact changed',name)
    return sha(OUT/'control/PARITY.json')

def prepare(conditional=False):
    verify()
    arm='conditional' if conditional else 'control'
    dest=OUT/arm
    assert not dest.exists(),'Preserve previous attempt; register an implementation revision for any repair'
    control_sha=validated_control() if conditional else None
    dest.mkdir()
    try:
        local,ordinal,panel,info,m,tm,px,op,ref_m,ref_a,parity=context(conditional)
        print('Context ready '+arm,flush=True)
        cached,label_audit=monthly_features(local,ordinal,panel,m,tm)
        np.savez_compressed(dest/'TRAIN_INPUTS.npz',**cached)
        dump(dest/'MONTHLY_META.json',dict(monthly=m,training=tm))
        dump(dest/'LABEL_AUDIT.json',label_audit)
        checks={}
        if not conditional:
            with np.load(local.target.P/'TRAIN_INPUTS.npz') as ref:
                checks['monthly']=numeric_parity(cached,{k:ref[k] for k in ref.files})
        print('Monthly feature and label rebuild complete '+arm,flush=True)
        base=module('ptr169_rebuild_base',ROOT/'work/ptr169_rebuild_base.py')
        a,base_audit,weekly_data,weekly_meta=base.build_base(panel,m,px,op)
        np.savez_compressed(dest/'BASE_DATA.npz',**a)
        np.savez_compressed(dest/'WEEKLY_INPUTS.npz',**weekly_data)
        dump(dest/'WEEKLY_META.json',weekly_meta)
        dump(dest/'BASE_AUDIT.json',base_audit)
        np.testing.assert_array_equal(a['eligible'],cached['eligible'])
        a,model_audit=models(local,a,cached,m,tm)
        np.savez_compressed(dest/'DATA.npz',**a)
        dump(dest/'MODEL_AUDIT.json',model_audit)
        if not conditional:
            checks['incumbent_arrays']=numeric_parity(a,ref_a)
            assert weekly_meta==read(O/'monthly_state_rank_research/TRAIN_DATA.json'),'Weekly metadata parity'
            with np.load(O/'monthly_state_rank_research/TRAIN_DATA.npz') as ref:
                required={'X','Y','adverse','good','past_score','offsets','event'}
                assert set(weekly_data)==required,'Rebuilt weekly keys differ'
                assert required.issubset(ref.files),'Reference weekly inputs incomplete'
                checks['weekly']=numeric_parity(weekly_data,{k:ref[k] for k in required})
        policy=read(local.P/'INCUMBENT_POLICY.json')
        chosen,score=local.choices(local.configs()[4],a,m,122,policy=policy)
        orders=local.rp.make_orders(chosen,score,m)
        if not conditional:
            ref_chosen,ref_score=local.choices(local.configs()[4],ref_a,ref_m,122,policy=policy)
            assert chosen==ref_chosen,'Control ordered selection mismatch'
            numeric_parity({'score':score},{'score':ref_score})
            assert orders==read(local.P/'local_linear_ORDERS.json'),'Control order mismatch'
        dump(dest/'META.json',dict(monthly=m,training=tm,weekly=weekly_meta))
        dump(dest/'ORDERS.json',orders)
        dump(dest/'AUDIT.json',dict(raw_input_sha256=info['source_hashes'],panel_parity=parity,
            labels=label_audit,base=base_audit,models=model_audit))
        verify()
        if conditional:assert validated_control()==control_sha,'Control receipt changed during reconstruction'
        dump(dest/'PARITY.json',dict(passed=True,arm=arm,checks=checks,
            original_control_required=conditional,returns_computed=False,
            control_parity_sha256=control_sha,
            artifacts_sha256={p.name:sha(p) for p in sorted(dest.iterdir()) if p.is_file()},
            source_plan_sha256=sha(OUT/'PLAN.json'),completed_utc=stamp()))
        print(json.dumps(dict(status='PREPARED',arm=arm,orders=len(orders),returns_computed=False)),flush=True)
    except Exception as exc:
        dump(dest/'FAILURE.json',dict(created_utc=stamp(),type=type(exc).__name__,message=str(exc),
             plan_sha256=sha(OUT/'PLAN.json'),returns_computed=False))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['register','control','conditional'])
    args=parser.parse_args()
    if args.phase=='register':register()
    else:prepare(args.phase=='conditional')
