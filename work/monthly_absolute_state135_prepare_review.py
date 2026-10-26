"""Independent round135 prepare audit; no data access or new fitting on import.

Only the frozen134 certificate is reused.  Absolute-state classification is
implemented here, independently of the new producer, before the frozen
independent scalar score/episode oracle is applied.
"""
from pathlib import Path
from datetime import date
import argparse
import copy
import hashlib
import json
import math
import numbers
import re
import sys
import numpy as np
import pandas as pd
import monthly_latent_state134_prepare_review as old

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'outputs/monthly_absolute_state135'
PRIOR = 'outputs/monthly_latent_state134'
NAMES, CACHE_FILES, CHECKS = old.NAMES, old.CACHE_FILES, old.CHECKS
INPUTS = old.INPUTS + ('reused_plan','reused_cache','reused_state','reused_data',
                      'reused_prepare','reused_completion','reused_outcome')
PRIOR_PLAN_SHA = 'a5c762d170f1b8bbee493411f2a85b171160e2a19ece703cd047827edb7bcbee'
PRIOR_PREPARE_SHA = '5feef213fa29fce1a58867dc870f40b3af2284e04573f3364ed45dcbde5952f3'
PROPOSAL_SHA = '5bf4f3c14d7fceceb65c46ef29b5861058fa89d456b23a0e80fe8a339b3530a7'
PRIOR_COMPLETION_SHA = 'c3ce897a194112ac3b840f620bc9f7d48bb09a54d26fd0b1b69437a039df4cc6'
PRIOR_ORACLE_SHA = 'd8d43226a7858f86f24ba2d8d6ec518daace404c77edf75b3714b3f61d63e06e'
require = old.require


def number(value):
    require(isinstance(value, numbers.Real) and not isinstance(value,(bool,np.bool_)), 'Real number required')
    out=float(value); require(math.isfinite(out), 'Finite number required'); return out


def state_classification(model):
    require(isinstance(model,dict) and model.get('available') is True and model.get('status')=='available'
            and model.get('reason') is None and model.get('fit_attempted') is True, 'Available fitted model')
    certificate=model.get('certificate',{})
    require(all(certificate.get(k) is True for k in ('converged','all_arithmetic_finite','low_variance_state_first')),
            'Certified source model')
    scaler=model.get('scaler'); p=model.get('parameters')
    require(isinstance(scaler,dict) and isinstance(p,dict), 'Scaler and parameters')
    number(scaler.get('mean')); sd=number(scaler.get('sd')); require(sd>0, 'Positive scale')
    require(len(p.get('means',[]))==2, 'Two state means')
    for x in p['means']: number(x)
    v=p.get('variances',[]);require(len(v)==2, 'Two state variances')
    lo,hi=map(number,v);require(0<lo<hi, 'Ordered positive state variances')
    require(len(p.get('transition',[]))==2, 'Two transition rows')
    for row in [p.get('initial',[]),model.get('filtered_posterior',[])]+list(p['transition']):
        require(len(row)==2, 'Two probability entries')
        values=[number(x) for x in row]
        require(all(0<=x<=1 for x in values) and abs(math.fsum(values)-1)<=1e-12, 'Normalized probabilities')
    sigma=[sd*math.sqrt(v)*math.sqrt(252) for v in (lo,hi)]
    require(all(math.isfinite(x) and x>0 for x in sigma), 'Representable state volatility')
    mask=[x<.10 for x in sigma];require(mask!=[False,True], 'Ordered quiet classes')
    return sigma,mask


def mapping_oracle(audit, filtered, memoryless):
    require(isinstance(audit,dict) and 'reused_model_fits' not in audit, 'Original134 audit')
    years,signals=audit.get('years'),audit.get('signals')
    require(isinstance(years,list) and isinstance(signals,list), 'Year/signal lists')
    count=audit.get('new_model_fits');require(type(count) is int and count>=0, 'Source fit count')
    arrays=[]
    for array in (filtered,memoryless):
        x=np.asarray(array);require(x.shape==(len(signals),) and x.dtype.kind in 'fiu', 'Aligned numeric probability array')
        require(not np.isinf(x).any(), 'Infinite probability');arrays.append(x)
    out=copy.deepcopy(audit);out.update(new_model_fits=0,reused_model_fits=count)
    byyear={};previous=0
    for row,target in zip(years,out['years']):
        y=row.get('year');require(type(y) is int and previous<y<=9999, 'Unique ordered year')
        require(type(row.get('fit_attempted')) is bool, 'Boolean fit attempt')
        require(row.get('status') in ('available','unavailable'), 'Year status')
        if row['status']=='available':
            require(row['fit_attempted'] and row.get('reason') is None, 'Available year metadata')
            sigma,mask=state_classification(row.get('model'))
        else:
            require(isinstance(row.get('reason'),str) and bool(row['reason']), 'Unavailable reason')
            m=row.get('model');require(m is not None or not row['fit_attempted'], 'Attempted fit record')
            if m is not None:
                require(isinstance(m,dict) and m.get('available') is False and m.get('status')=='unavailable'
                        and m.get('fit_attempted') is row['fit_attempted'], 'Unavailable model metadata')
            sigma=mask=None
        target.update(annualized_state_volatility=sigma,quiet_state_mask=mask)
        byyear[y]=mask;previous=y
    require(count==sum(x['fit_attempted'] for x in years), 'Source fit count matches years')
    values=[np.full(len(signals),np.nan),np.full(len(signals),np.nan)];last=None
    for i,(row,target) in enumerate(zip(signals,out['signals'])):
        require(type(row.get('signal_index')) is int and row['signal_index']==i, 'Ordered index')
        d=row.get('signal_date');require(isinstance(d,str), 'ISO signal date')
        day=date.fromisoformat(d);require(str(day)==d and (last is None or day>last), 'Unique ordered signal date')
        y=row.get('year');require(type(y) is int and y==day.year and y in byyear, 'Signal year binding')
        available=row.get('available');require(type(available) is bool, 'Boolean availability')
        require(row.get('reason') is None if available else isinstance(row.get('reason'),str) and bool(row['reason']), 'Signal reason')
        if available:require(byyear[y] is not None, 'Available signal requires available year')
        for j,k in enumerate(('filtered_probability','memoryless_probability')):
            v=row.get(k)
            if available:
                v=number(v);require(0<=v<=1 and v==arrays[j][i], 'Exact source array/audit probability')
                mask=byyear[y];mapped=1. if all(mask) else 0. if not any(mask) else v
                values[j][i]=mapped;target[k]=mapped
            else:
                require(k in row and v is None and math.isnan(float(arrays[j][i])), 'Exact source null/NaN mask')
            target['original_'+k]=v
        last=day
    return dict(filtered=values[0],memoryless=values[1],audit=out)


def source_files(root=None):
    root=ROOT if root is None else Path(root).resolve()
    return sorted(set([root/'work/monthly_absolute_state135_prepare_review.py',
                       root/'work/test_monthly_absolute_state135_prepare_review.py']+old.source_files(root)))


def require_bound(reader, mapping, pins, base=None):
    verified=reader.verify(mapping,base)
    require(all(pins.get(k)==v for k,v in verified.items()), 'Unregistered reused dependency')
    return verified


def reuse_certificate(reader,pins,inputs):
    directory=reader.path(PRIOR)
    expected=dict(reused_plan='PLAN.json',reused_cache='CACHE.json',reused_state='STATE_AUDIT.json',
                  reused_data='DATA.npz',reused_prepare='INDEPENDENT_PREPARE.json',
                  reused_completion='COMPLETION.json',reused_outcome='INDEPENDENT_OUTCOME.json')
    for k,file in expected.items():require(inputs[k]==directory/file, 'Frozen134 input wiring '+k)
    require(reader.sha(inputs['reused_plan'])==PRIOR_PLAN_SHA, 'Frozen134 PLAN identity')
    require(reader.sha(inputs['reused_prepare'])==PRIOR_PREPARE_SHA, 'Frozen134 independent prepare identity')
    plan=reader.json(inputs['reused_plan']);require(plan['round']==134 and plan['arms']==list(NAMES), 'Frozen134 policy')
    require_bound(reader,plan['hashes'],pins)
    cache=reader.json(inputs['reused_cache']);require(set(cache)==set(CACHE_FILES), 'Frozen134 cache shape')
    require_bound(reader,cache,pins,directory)
    require(reader.sha(inputs['reused_completion'])==PRIOR_COMPLETION_SHA, 'Frozen134 completion identity')
    completion=reader.json(inputs['reused_completion']);require(completion.get('round')==134 and completion.get('status')=='completed_independently_verified', 'Frozen134 completion')
    artifacts=require_bound(reader,{v['path']:v['sha256'] for v in completion['artifacts'].values()},pins)
    for file in (*CACHE_FILES,'PLAN.json','CACHE.json','INDEPENDENT_PREPARE.json','INDEPENDENT_OUTCOME.json'):
        require(artifacts.get(str(directory/file))==reader.sha(directory/file),'Completed134 artifact '+file)
    for key,stage in (('reused_prepare','prepare'),('reused_outcome','outcome')):
        receipt=reader.json(inputs[key]);require(receipt.get('passed') is True, 'Passed134 receipt')
        require(receipt.get('plan_sha256')==PRIOR_PLAN_SHA and receipt.get('cache_sha256')==reader.sha(inputs['reused_cache'])
                and receipt.get('orders_sha256')==reader.sha(directory/'PRETRAIN_ORDERS.json'), '134 receipt binding')
        require_bound(reader,receipt['source_hashes'],pins);require_bound(reader,receipt['artifact_hashes'],pins)
        if stage=='prepare':
            require(receipt.get('stage')=='prepare' and all(receipt.get(k) is True for k in CHECKS), '134 prepare guarantees')
            require(receipt.get('producer_modules_imported') is False and receipt.get('returns_or_nav_computed') is False
                    and type(receipt.get('monthly_label_models_fit')) is int and receipt['monthly_label_models_fit']==0, '134 isolation')
    return plan,cache


def audit(root,plan_path,plan_sha):
    require(re.fullmatch('[0-9a-f]{64}',plan_sha) is not None, 'Explicit PLAN SHA')
    r=old.Reader(root);p=r.path(STUDY);require(r.path(plan_path)==p/'PLAN.json', 'Dedicated PLAN')
    require(r.sha(plan_path)==plan_sha, 'PLAN SHA');plan=r.json(plan_path)
    require(plan.get('round')==135 and plan.get('status')=='REGISTERED' and plan.get('arms')==list(NAMES)
            and plan.get('primary_challenger')==NAMES[2], 'Round135 policy')
    require(r.path(plan['output_directory'])==p, 'Output directory')
    require(plan['runtime']==dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__), 'Runtime')
    require(not any((p/x).exists() for x in ('TRAIN_TRIALS.json','FROZEN.json','RESULTS.json')), 'Before outcome stages')
    pins=r.verify(plan['hashes']);require(set(plan['inputs'])==set(INPUTS), 'Fourteen named inputs')
    inputs={}
    for k,ref in plan['inputs'].items():
        path=r.path(ref['path']);require(r.sha(path)==ref['sha256']==pins.get(str(path)), 'Bound input '+k);inputs[k]=path
    require(r.sha(inputs['proposal'])==PROPOSAL_SHA, 'Frozen135 proposal')
    require(all(plan.get(k)==0 for k in ('actual_model_fits_before_registration','actual_feature_calculations_before_registration',
                                       'actual_returns_before_registration')), 'Prospective registration')
    require(plan['tolerances'].get('scores_atol')==1e-10 and plan['tolerances'].get('scores_rtol')==0, 'Score tolerance')
    require(r.sha(r.path('work/monthly_latent_state134_prepare_review.py'))==PRIOR_ORACLE_SHA, 'Frozen independent oracle')
    sources={str(f):r.sha(f) for f in source_files(root)}
    require(r.verify(plan['independent_prepare_source_hashes'])==sources and all(pins.get(k)==v for k,v in sources.items()), 'Independent closure')
    reused_plan,_=reuse_certificate(r,pins,inputs)
    for k in old.INPUTS:
        if k!='proposal':require(inputs[k]==r.path(reused_plan['inputs'][k]['path']), 'Unchanged original input '+k)
    raw=require_bound(r,plan['raw_source_hashes'],pins)
    require(raw==require_bound(r,reused_plan['raw_source_hashes'],pins), 'Frozen raw sources')
    cache=r.json(p/'CACHE.json');require(set(cache)==set(CACHE_FILES), 'Seven cached outputs');r.verify(cache,p)
    meta=r.json(p/'META.json');require(meta==r.json(inputs['meta'])==r.json(r.path(PRIOR)/'META.json'), 'Exact original metadata')
    require(len(meta['dates'])==len(meta['symbols'])==122 and meta['train_months']==88 and meta['early_months']==53
            and len(meta['etfs'])==17 and all(len(x)==49 for x in meta['symbols']), 'Original122x49')
    before=r.npz(inputs['reused_data']);a=r.npz(p/'DATA.npz');require(len(before)==31 and set(a)==set(before), 'Original31 arrays')
    for k,v in before.items():
        if k not in ('latent_filtered','latent_memoryless'):
            require(a[k].shape==v.shape and a[k].dtype==v.dtype and a[k].tobytes()==v.tobytes(), 'Unchanged original array '+k)
    state=r.json(inputs['reused_state']);mapped=mapping_oracle(state,before['latent_filtered'],before['latent_memoryless'])
    require(r.json(p/'STATE_AUDIT.json')==mapped['audit'], 'Exact independently mapped audit and retained model evidence')
    for key,name in [('latent_filtered','filtered'),('latent_memoryless','memoryless')]:
        require(a[key].dtype==mapped[name].dtype and a[key].shape==mapped[name].shape
                and a[key].tobytes()==mapped[name].tobytes(), 'Exact mapped probabilities '+key)
    raw_rank,risk_rank=a['latent_raw_rank'],a['latent_risk_rank']
    scores,choices,_=old.score_oracle(a,meta,raw_rank,risk_rank,mapped['filtered'],mapped['memoryless'])
    saved_scores=r.npz(p/'SCORES.npz');saved_orders=r.json(p/'PRETRAIN_ORDERS.json')
    require(set(saved_scores)==set(saved_orders)==set(NAMES), 'Three fixed arms')
    for name in NAMES:
        expected=[dict(date=day,signal_date=day,end=meta['ends'][i],symbols=[meta['symbols'][i][j] for j in choice],
            scores=[float(scores[name][i,j]) for j in choice]) for i,(day,choice) in enumerate(zip(meta['dates'],choices[name]))]
        old.compare(saved_orders[name],expected,'orders/'+name)
        np.testing.assert_allclose(saved_scores[name],scores[name],rtol=0,atol=1e-10)
    old.original_helper().assert_exact_tree(saved_orders[NAMES[0]],r.json(inputs['original_orders']))
    require(saved_scores[NAMES[0]].tobytes()==scores[NAMES[0]].tobytes(), 'Baseline score bytes')
    for n in (0,1,36,53,60,76,88,122):
        prefix=copy.deepcopy(state);prefix['signals']=prefix['signals'][:n]
        allowed={x['year'] for x in prefix['signals']};prefix['years']=[x for x in prefix['years'] if x['year'] in allowed]
        prefix['new_model_fits']=sum(x['fit_attempted'] for x in prefix['years'])
        pm=mapping_oracle(prefix,before['latent_filtered'][:n],before['latent_memoryless'][:n])
        require(pm['audit']['signals']==mapped['audit']['signals'][:n], 'Mapping prefix isolation')
        ss,cc,_=old.score_oracle(a,meta,raw_rank,risk_rank,pm['filtered'],pm['memoryless'],n)
        for name in NAMES:require(cc[name]==choices[name][:n] and ss[name].tobytes()==scores[name][:n].tobytes(), 'Score/order prefix')
    missing=np.full(122,np.nan);ss,cc,_=old.score_oracle(a,meta,raw_rank,risk_rank,missing,missing)
    for name in NAMES[1:]:require(ss[name].tobytes()==scores[NAMES[0]].tobytes() and cc[name]==choices[NAMES[0]], 'Missing baseline fallback')
    parity=r.json(p/'PRICE_PARITY.json');require(parity==r.json(r.path(PRIOR)/'PRICE_PARITY.json'), 'Frozen price parity')
    feature=r.json(p/'FEATURE_AUDIT.json')
    expected_feature=dict(original_arrays_unchanged=True,reused_completed_round134=True,original_months=122,original_width=49,
        new_model_fits=0,reused_model_fits=state['new_model_fits'],returns_or_nav_computed=False,
        original_auxiliary_states_and_core_coefficients_preserved=True)
    old.compare(feature,expected_feature,'FEATURE_AUDIT')
    required={str(p/x):r.sha(p/x) for x in (*CACHE_FILES,'PLAN.json','CACHE.json')};allowed=dict(pins,**required)
    for key,value in list(r.seen.items()):require(r.sha(key)==value and allowed.get(key)==value, 'Unbound consumed artifact '+key)
    return dict(schema_version=1,passed=True,stage='prepare',plan_sha256=plan_sha,cache_sha256=r.sha(p/'CACHE.json'),
        orders_sha256=r.sha(p/'PRETRAIN_ORDERS.json'),**{k:True for k in CHECKS},source_hashes=sources,artifact_hashes=dict(r.seen),
        producer_modules_imported=False,returns_or_nav_computed=False,monthly_label_models_fit=0,independent_daily_model_fits=0,
        reused_model_fits=state['new_model_fits'],orders_checked=366,all_signal_prefixes_verified=True,
        frozen134_receipt_and_all_artifacts_verified=True,calendar_overlay_independently_verified=True,
        zero_posterior_is_valid_not_missing=True,missing_posterior_exact_baseline=True,
        limitations=['No HMM refit: original134 numerical, price and calendar work is reused through its complete verified certificate.',
                     'Absolute classification is state-conditional daily volatility annualized by sqrt252, not a multi-day Markov risk forecast.',
                     'Proxy universe, current-vintage prices and reused historical sample limitations remain.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan-sha',required=True);p.add_argument('--prepare-complete',action='store_true')
    p.add_argument('--plan',type=Path,default=ROOT/STUDY/'PLAN.json');p.add_argument('--output',type=Path,default=ROOT/STUDY/'INDEPENDENT_PREPARE.json')
    args=p.parse_args();require(args.prepare_complete, 'Only after registered preparation')
    result=audit(ROOT,args.plan,args.plan_sha)
    with args.output.open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,output=str(args.output),sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__=='__main__':main()
