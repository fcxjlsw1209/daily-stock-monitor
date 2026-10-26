"""Independent round142 prepare audit; actual cache reads occur only in audit().

Scenario arithmetic and joint selection are implemented here, not imported from
round142 producers. Synthetic tests do not open actual feature/order/return files.
"""
from pathlib import Path
from itertools import combinations
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import re
import sys
import numpy as np
import pandas as pd
import monthly_price_delay140_prepare_review as prior

old=prior.old
ROOT=Path(__file__).resolve().parents[1]
STUDY='outputs/monthly_formation142'
LOCAL=old.LOCAL
PROPOSAL_SHA='13d6f8c27ff86a6a81fa54bcb53012db49163a95c98da2c7f8aeec80463f437b'
CALENDAR_SHA=prior.CALENDAR_SHA
NAMES=('incumbent_local67','formation_common_delete_mean','formation_common_delete_worst')
INPUTS=prior.INPUTS
CHECKS=('original27_arrays_byte_identical','all_features_independently_reconstructed',
        'all_three_score_matrices_and_ordered_choices_independently_verified',
        'common_scenario_selector_and_no_double_episode_verified','zero_and_missing_exact_fallback',
        'raw_prices_independently_verified','calendar_overlay_independently_verified',
        'all_signal_prefixes_verified','formation_block_deletions_independently_verified')
EXTRA=('formation_rank','formation_count','formation_ready','formation_valid')
CACHE_FILES=prior.CACHE_FILES
ATOL=1e-12
require=old.require
canonical_axis=prior.canonical_axis
compare=prior.compare


def source_files(root=None):
    root=ROOT if root is None else Path(root).resolve()
    return sorted(set([root/'work/monthly_formation142_prepare_review.py',
                       root/'work/test_monthly_formation142_prepare_review.py']+prior.source_files(root)))


def rank_oracle(raw):
    """Pandas average-rank pct convention; unavailable entries are neutral zero."""
    values=list(raw);require(not any(math.isinf(float(v)) for v in values),'Infinite rank input')
    finite=[float(v) for v in values if math.isfinite(float(v))]
    answer=[]
    for v in values:
        if not math.isfinite(float(v)):answer.append(0.);continue
        lower=sum(x<v for x in finite);equal=sum(x==v for x in finite)
        answer.append((lower+(equal+1)/2)/len(finite)-.5)
    return np.asarray(answer,dtype=float)


def sigma_oracle(values):
    """Independent scalar, two-pass sample standard deviation of 63 returns."""
    v=[float(x) for x in values];require(len(v)==64,'Exactly64 volatility prices')
    require(not any(math.isinf(x) for x in v),'Infinite risk-window price')
    if not all(math.isfinite(x) and x>0 for x in v):return float('nan')
    rr=[v[k]/v[k-1]-1. for k in range(1,64)]
    require(all(math.isfinite(x) for x in rr),'Nonfinite risk return')
    mean=math.fsum(rr)/63
    variance=math.fsum((x-mean)**2 for x in rr)/62
    require(math.isfinite(variance),'Nonfinite risk variance')
    result=math.sqrt(variance)
    return result if result>0 else float('nan')


def positive_ratio(a,b,where):
    value=float(a)/float(b)
    if not math.isfinite(value) or value<=0:raise FloatingPointError('Nonpositive or nonfinite '+where)
    return value


def scenario_oracle(prices,signal_index,*,calm):
    data=np.asarray(prices)
    require(data.ndim==2 and data.shape[1]>0,'Price matrix')
    require(isinstance(signal_index,(int,np.integer)) and not isinstance(signal_index,(bool,np.bool_))
            and 0<=signal_index<len(data),'Signal index')
    require(isinstance(calm,(bool,np.bool_)),'Boolean calm state')
    horizon=252 if calm else 294;weight=.5 if calm else .52;width=data.shape[1]
    result=dict(ready=False,reason='insufficient_formation_history',core_weight=weight,
        horizon=horizon,skip=42,blocks=[],core_ranks=None,scenario_core_ranks=None,original_finite_mask=None)
    if signal_index<horizon:return result
    first,last=signal_index-horizon,signal_index-42
    formation=np.asarray(data[first:last+1],dtype=float)
    require(not np.isinf(formation).any(),'Infinite formation price')
    sigmas=[1. if calm else sigma_oracle(data[signal_index-63:signal_index+1,j]) for j in range(width)]
    valid=[];gross=[];raw=[];complete=[]
    for j in range(width):
        path=formation[:,j];ends=path[[0,-1]]
        good=all(math.isfinite(x) and x>0 for x in ends)
        total=positive_ratio(ends[-1],ends[0],'formation gross') if good else float('nan')
        ok=good and math.isfinite(sigmas[j]) and sigmas[j]>0
        value=(total-1.)/sigmas[j] if ok else float('nan')
        if ok and not math.isfinite(value):raise FloatingPointError('Nonfinite original core')
        valid.append(ok);gross.append(total);raw.append(value)
        complete.append(all(math.isfinite(x) and x>0 for x in path))
    result['original_finite_mask']=np.asarray(valid,dtype=bool)
    result['core_ranks']=rank_oracle(raw)
    if any(v and not c for v,c in zip(valid,complete)):
        result['reason']='incomplete_originally_measurable_formation';return result
    matrices=[]
    for offset in range(0,horizon-42,21):
        values=[]
        for j,ok in enumerate(valid):
            if not ok:values.append(float('nan'));continue
            block=positive_ratio(formation[offset+21,j],formation[offset,j],'deleted-block gross')
            value=(gross[j]/block-1.)/sigmas[j]
            if not math.isfinite(value):raise FloatingPointError('Scenario changed original finite mask')
            values.append(value)
        require(np.array_equal(np.isfinite(values),valid),'Scenario finite mask')
        matrices.append(rank_oracle(values))
        result['blocks'].append(dict(start_price_index=first+offset,end_price_index=first+offset+21,deleted_returns=21))
    result.update(ready=True,reason='ready',scenario_core_ranks=np.asarray(matrices,dtype=float))
    return result


def features_oracle(panel,meta,original,sessions):
    prices=panel['adj'];require(isinstance(prices,pd.DataFrame) and prices.columns.is_unique,'Unique adjusted price frame')
    axis=canonical_axis(prices.index);calendar=canonical_axis(sessions);dates=canonical_axis(meta['dates'])
    n=len(dates);require(n==len(meta['symbols']),'Month axes')
    require(len(meta['etfs'])==len(set(meta['etfs']))==17,'Original17 ETFs')
    arrays=dict(formation_rank=np.full((n,12,49),np.nan),formation_count=np.zeros(n,np.int64),
        formation_ready=np.zeros(n,bool),formation_valid=np.zeros((n,49),bool));months=[]
    for i,(day,names) in enumerate(zip(dates,meta['symbols'])):
        require(len(names)==len(set(names))==49 and set(meta['etfs'])<=set(names),'Original49 names')
        require(day in axis and day in calendar,'Signal date in source and official calendar')
        t=axis.get_loc(day);require(isinstance(t,(int,np.integer)) and t>=294,'Original core history')
        needed=axis[t-294:t+1]
        require(needed.equals(calendar[(calendar>=needed[0])&(calendar<=day)]),'Missing official formation sessions')
        # Two independent paths: original pandas raw formula versus scalar
        # volatility/deletion reconstruction, including masks before ranking.
        p=prices.loc[:day,names].astype(float);recent=p.iloc[-64:].pct_change(fill_method=None)
        sigma=recent.iloc[1:].std(ddof=1,skipna=False).replace(0,np.nan)
        require(np.asarray(original['state_calm10']).dtype.kind=='b','Original state boolean')
        calm=bool(original['state_calm10'][i]);weight=.5 if calm else .52
        spy_sigma=sigma_oracle(prices.loc[:day,'SPY'].iloc[-64:].to_numpy())
        require(math.isfinite(spy_sigma),'Finite market state volatility')
        require(calm==(spy_sigma*math.sqrt(252)<.10),'Original calm state parity')
        require(np.all(np.asarray(original['core_weight'][i])==weight),'Original core weight parity')
        numerator=p.iloc[-43]/p.iloc[-253 if calm else -295]-1.
        raw=numerator if calm else numerator/sigma
        require(not np.isinf(raw.to_numpy()).any(),'Infinite original raw core')
        finite=np.isfinite(raw.to_numpy());ranks=rank_oracle(raw.to_numpy())
        np.testing.assert_allclose(weight*ranks,original['core'][i],rtol=0,atol=ATOL)
        eligible=np.asarray(original['eligible'][i]);require(eligible.dtype.kind=='b' and eligible.shape==(49,),'Original eligibility schema')
        ends=p.iloc[[-43,-253 if calm else -295]].to_numpy()
        require(np.all(np.isfinite(ends[:,eligible])&(ends[:,eligible]>0)),'Eligible formation endpoints must be positive finite')
        result=scenario_oracle(p.to_numpy(),len(p)-1,calm=calm)
        require(np.array_equal(result['original_finite_mask'],finite),'Original raw finite mask parity')
        np.testing.assert_allclose(result['core_ranks'],ranks,rtol=0,atol=ATOL)
        count=len(result['blocks']);arrays['formation_count'][i]=count
        arrays['formation_ready'][i]=result['ready'];arrays['formation_valid'][i]=finite
        if result['ready']:arrays['formation_rank'][i,:count]=result['scenario_core_ranks']
        months.append(dict(signal_index=i,date=meta['dates'][i],calm=calm,horizon=result['horizon'],
            ready=result['ready'],reason=result['reason'],blocks=result['blocks'],
            original_finite_mask=finite.tolist(),selectors={}))
    return dict(arrays=arrays,audit=dict(months=months,original_months=n,original_width=49,new_model_fits=0,
        monthly_label_models_fit=0,returns_or_nav_computed=False,original_arrays_unchanged=True,
        raw_core_mask_and_rank_verified=True))


def selector_oracle(pre,chosen,core,weight,scenarios,eligible,keys,*,arm,history_available=True):
    """Enumerate trios using scalar objectives; never run old episode here."""
    require(arm in ('mean_scenario','worst_scenario'),'Selector arm')
    pre=np.asarray(pre);core=np.asarray(core);eligible=np.asarray(eligible);keys=list(keys);chosen=list(chosen)
    width=len(keys)
    require(width>0 and len(set(keys))==width and all(type(k)is str and k for k in keys),'Unique keys')
    require(pre.shape==core.shape==eligible.shape==(width,) and eligible.dtype.kind=='b','Selector axes')
    require(pre.dtype.kind in 'fiu' and core.dtype.kind in 'fiu' and np.isfinite(pre).all() and np.isfinite(core).all(),'Finite original scores')
    require(not isinstance(weight,(bool,np.bool_)) and weight in (.5,.52),'Original core weight')
    require(len(chosen)<=3 and len(set(chosen))==len(chosen) and all(isinstance(j,(int,np.integer)) and not isinstance(j,(bool,np.bool_)) and 0<=j<width and eligible[j] for j in chosen),'Original choice')
    history=np.asarray(history_available)
    require(history.dtype.kind=='b' and history.shape in ((),(width,)),'History flags')
    if history.ndim==0:history=np.full(width,bool(history),bool)
    order=lambda j:(-float(pre[j]),keys[j])
    candidates=sorted(set(sorted(np.flatnonzero(eligible),key=order)[:8])|set(chosen),key=order)
    scenarios=None if scenarios is None else np.asarray(scenarios)
    if scenarios is not None:
        require(scenarios.ndim==2 and scenarios.shape[1]==width and scenarios.dtype.kind in 'fiu' and not np.isinf(scenarios).any(),'Scenario schema')
        usable=scenarios[np.isfinite(scenarios)];require((np.abs(usable)<=.5).all(),'Rank bounds')
    count=0 if scenarios is None else len(scenarios)
    def out(indices,objective,fallback,reason):
        return dict(arm=arm,selected_indices=[int(j) for j in indices],selected_keys=[keys[j] for j in indices],
            candidate_indices=[int(j) for j in candidates],candidate_keys=[keys[j] for j in candidates],
            scenario_count=count,objective=objective,fallback=fallback,reason=reason)
    if len(chosen)<3:return out(chosen,None,True,'original_fewer_than_three')
    if scenarios is None:return out(chosen,None,True,'missing_scenarios')
    if count==0:return out(chosen,None,True,'empty_scenarios')
    if np.isnan(scenarios).any():return out(chosen,None,True,'nan_scenario_values')
    if not all(history[j] for j in candidates):return out(chosen,None,True,'candidate_history_unavailable')
    weighted=weight*scenarios
    if np.all(np.abs(weighted-core)<=ATOL):return out(chosen,None,True,'unchanged_weighted_core')
    adjusted=pre[None,:]+weighted-core[None,:]
    require(np.isfinite(adjusted).all(),'Finite adjusted scenario scores')
    entries=[]
    for combo in combinations(candidates,3):
        portfolio=[math.fsum(float(row[j]) for j in combo)/3 for row in adjusted]
        objective=math.fsum(portfolio)/count if arm=='mean_scenario' else min(portfolio)
        entries.append((objective,math.fsum(float(pre[j]) for j in combo),tuple(sorted(keys[j]for j in combo)),combo))
    high=max(e[0]for e in entries);tied=[e for e in entries if high-e[0]<=ATOL]
    keep=[e for e in tied if set(e[3])==set(chosen)]
    winner=keep[0] if keep else sorted(tied,key=lambda e:(-e[1],e[2]))[0]
    return out(sorted(winner[3],key=order),float(winner[0]),False,'selected_by_'+arm)


def score_oracle(a,meta,feature_arrays,n=None):
    h=old.original_helper();x=h.original_prefix(a,meta,n);n=x['n'];width=x['core'].shape[1]
    shapes={'formation_rank':(n,12,width),'formation_count':(n,), 'formation_ready':(n,), 'formation_valid':(n,width)}
    f={k:np.asarray(feature_arrays[k])[:n] for k in EXTRA}
    for key in EXTRA:require(f[key].shape==shapes[key],'Feature axes '+key)
    require(f['formation_rank'].dtype==np.dtype(float) and f['formation_count'].dtype==np.dtype(np.int64)
        and f['formation_ready'].dtype==f['formation_valid'].dtype==np.dtype(bool),'Feature dtypes')
    chosen,final,_=h.episode(x['local'],x)
    scores={NAMES[0]:final};choices={NAMES[0]:chosen};details={}
    for name,arm in zip(NAMES[1:],('mean_scenario','worst_scenario')):
        scores[name]=x['local'].copy();choices[name]=[];details[name]=[]
        for i in range(n):
            count=int(f['formation_count'][i]);ready=bool(f['formation_ready'][i])
            required=10 if x['calm'][i] else 12
            require(count==(required if ready else 0),'Count agrees with readiness/state')
            require(np.isnan(f['formation_rank'][i,count:]).all(),'Unused scenarios NaN')
            row=f['formation_rank'][i,:count] if ready else None
            if ready:require(np.isfinite(row).all(),'Ready scenarios finite')
            item=selector_oracle(x['local'][i],chosen[i],x['core'][i],float(x['cw'][i,0]),
                row,x['eligible'][i],x['names'][i],arm=arm,
                history_available=f['formation_valid'][i] if ready else False)
            choices[name].append(item['selected_indices']);details[name].append(item)
            if item['fallback']:scores[name][i]=final[i]
    return scores,choices,details


def audit(root,plan_path,plan_sha):
    require(re.fullmatch('[0-9a-f]{64}',plan_sha) is not None,'Explicit PLAN SHA')
    r=old.Reader(root);p=r.path(STUDY);require(r.path(plan_path)==p/'PLAN.json','Dedicated PLAN')
    require(r.sha(plan_path)==plan_sha,'PLAN SHA');plan=r.json(plan_path)
    require(plan.get('round')==142 and plan.get('status')=='REGISTERED' and plan.get('arms')==list(NAMES)
            and plan.get('primary_challenger')==NAMES[2],'Round142 policy')
    require(r.path(plan['output_directory'])==p,'Output directory')
    require(plan['runtime']==dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),'Runtime')
    require(not any((p/x).exists() for x in ('TRAIN_TRIALS.json','FROZEN.json','RESULTS.json')),'Before outcome stages')
    pins=r.verify(plan['hashes']);require(set(plan['inputs'])==set(INPUTS),'Named inputs')
    inputs={}
    for key,ref in plan['inputs'].items():
        path=r.path(ref['path']);require(r.sha(path)==ref['sha256']==pins.get(str(path)),'Bound input '+key);inputs[key]=path
    require(r.sha(inputs['proposal'])==PROPOSAL_SHA,'Final proposal')
    require(all(type(plan.get(k)) is int and plan[k]==0 for k in ('actual_model_fits_before_registration',
        'actual_feature_calculations_before_registration','actual_returns_before_registration')),'Prospective registration')
    require(plan['tolerances'].get('feature_atol')==ATOL and plan['tolerances'].get('feature_rtol')==0
            and plan['tolerances'].get('scores_atol')==ATOL and plan['tolerances'].get('scores_rtol')==0,'Frozen tolerances')
    sources={str(f):r.sha(f) for f in source_files(root)}
    require(r.verify(plan['independent_prepare_source_hashes'])==sources and all(pins.get(k)==v for k,v in sources.items()),'Independent closure')
    oldplan=r.json(LOCAL+'/PLAN.json');raw=r.verify(plan['raw_source_hashes'])
    require(raw==r.verify(oldplan['raw_source_hashes']) and len(raw)==64 and all(pins.get(k)==v for k,v in raw.items()),'Raw64 source identity')
    oldcache=r.verify(r.json(LOCAL+'/CACHE_HASHES.json'),r.path(LOCAL))
    require(all(pins.get(k)==v for k,v in oldcache.items()),'Original cache artifacts')
    for key,file in [('meta','META.json'),('policy','INCUMBENT_POLICY.json'),('original_orders','local_linear_ORDERS.json')]:
        require(inputs[key]==r.path(LOCAL+'/'+file),'Original wiring '+key)
    require(inputs['prices']==r.path(old.PRICES),'Original price input')
    h=old.original_helper();policy=r.json(inputs['policy']);require(all(policy.get(k)==v for k,v in h.POLICY.items()),'Frozen episode')
    cache=r.json(p/'CACHE.json');require(set(cache)==set(CACHE_FILES),'Six cache outputs');r.verify(cache,p)
    meta=r.json(p/'META.json');require(meta==r.json(inputs['meta']),'Exact metadata')
    require(len(meta['dates'])==len(meta['symbols'])==122 and meta['train_months']==88 and meta['early_months']==53
            and len(meta['etfs'])==17 and all(len(x)==49 for x in meta['symbols']),'Original122x49')
    original=r.npz(LOCAL+'/DATA.npz');a=r.npz(p/'DATA.npz')
    require(len(original)==27 and set(a)==set(original)|set(EXTRA),'31 array schema')
    for key,value in original.items():
        require(a[key].shape==value.shape and a[key].dtype==value.dtype and a[key].tobytes()==value.tobytes(),'Original array bytes '+key)
    require(r.sha(old.LOADER)==old.LOADER_SHA,'Raw loader code')
    spec=importlib.util.spec_from_file_location('formation142_raw_loader',r.path(old.LOADER))
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    panel,info=loader.load_panel(meta['etfs']);used=dict(info['source_hashes']);used[str(r.path(old.LOADER))]=old.LOADER_SHA
    require({str(r.path(k)):v for k,v in used.items()}==raw,'Actually consumed raw sources');r.verify(used)
    prices=r.npz(inputs['prices']);require(set(prices)=={'dates','symbols','close','opening'},'Price cache schema')
    parity=old.price_parity(panel,prices);old.compare(r.json(p/'PRICE_PARITY.json'),parity,'PRICE_PARITY')
    calendar_path=inputs['sessions'].parent/'CALENDAR_PLAN.json';require(r.sha(calendar_path)==CALENDAR_SHA,'Calendar plan hash')
    calendar_plan=r.json(calendar_path);calendar_bindings=r.verify(calendar_plan['bindings'])
    require(all(pins.get(k)==v for k,v in calendar_bindings.items()),'Calendar source closure')
    require(calendar_plan['start']=='2015-01-01' and calendar_plan['end']=='2027-12-31','Calendar coverage')
    sessions=old.official_sessions(r,calendar_plan['end']);sessions=sessions[sessions>=calendar_plan['start']]
    saved_sessions=r.json(inputs['sessions'])
    require(saved_sessions['plan_sha256']==CALENDAR_SHA and saved_sessions['sessions']==[str(d.date()) for d in sessions],'All official sessions and overlay')
    expected=features_oracle(panel,meta,original,sessions)
    for key in EXTRA:
        require(a[key].shape==expected['arrays'][key].shape and a[key].dtype==expected['arrays'][key].dtype,'Feature schema '+key)
        if key=='formation_rank':np.testing.assert_allclose(a[key],expected['arrays'][key],rtol=0,atol=ATOL,equal_nan=True)
        else:np.testing.assert_array_equal(a[key],expected['arrays'][key])
    scores,choices,details=score_oracle(original,meta,expected['arrays'])
    for i,row in enumerate(expected['audit']['months']):row['selectors']={name:details[name][i]for name in NAMES[1:]}
    compare(r.json(p/'FEATURE_AUDIT.json'),expected['audit'],'FEATURE_AUDIT')
    saved_scores=r.npz(p/'SCORES.npz');saved_orders=r.json(p/'PRETRAIN_ORDERS.json')
    require(set(saved_scores)==set(saved_orders)==set(NAMES),'Three output arms')
    for name in NAMES:
        wanted=[dict(date=day,signal_date=day,end=meta['ends'][i],symbols=[meta['symbols'][i][j] for j in choice],
            scores=[float(scores[name][i,j])for j in choice])for i,(day,choice)in enumerate(zip(meta['dates'],choices[name]))]
        compare(saved_orders[name],wanted,'orders/'+name)
        require(saved_scores[name].shape==scores[name].shape and saved_scores[name].dtype==scores[name].dtype,'Saved score schema')
        np.testing.assert_allclose(saved_scores[name],scores[name],rtol=0,atol=ATOL)
    h.assert_exact_tree(saved_orders[NAMES[0]],r.json(inputs['original_orders']))
    require(saved_scores[NAMES[0]].tobytes()==scores[NAMES[0]].tobytes(),'Baseline score bytes')
    # All123 possible prefixes independently rebuild source features after
    # truncating source prices at that prefix's last decision, then choices.
    for n in range(123):
        cut=panel['adj'].loc[:meta['dates'][n-1]] if n else panel['adj'].iloc[:0]
        subset=dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n])
        prefix=features_oracle(dict(panel,adj=cut),subset,original,sessions)
        for key in EXTRA:np.testing.assert_array_equal(prefix['arrays'][key],expected['arrays'][key][:n])
        ss,cc,dd=score_oracle(original,meta,prefix['arrays'],n)
        for i,row in enumerate(prefix['audit']['months']):row['selectors']={name:dd[name][i]for name in NAMES[1:]}
        compare(prefix['audit']['months'],expected['audit']['months'][:n],'Source prefix')
        for name in NAMES:
            require(cc[name]==choices[name][:n] and ss[name].tobytes()==scores[name][:n].tobytes(),'Score/choices prefix')
    missing={k:v.copy()for k,v in expected['arrays'].items()}
    missing['formation_rank'][:]=np.nan;missing['formation_count'][:]=0;missing['formation_ready'][:]=False
    unchanged={k:v.copy()for k,v in expected['arrays'].items()}
    for i in range(122):
        weight=float(np.asarray(original['core_weight'][i]).reshape(-1)[0]);count=10 if original['state_calm10'][i] else 12
        unchanged['formation_rank'][i]=np.nan;unchanged['formation_rank'][i,:count]=original['core'][i]/weight
        unchanged['formation_count'][i]=count;unchanged['formation_ready'][i]=True;unchanged['formation_valid'][i]=True
    for f in (missing,unchanged):
        ss,cc,_=score_oracle(original,meta,f)
        for name in NAMES[1:]:require(ss[name].tobytes()==scores[NAMES[0]].tobytes()and cc[name]==choices[NAMES[0]],'Missing/unchanged exact fallback')
    required={str(p/x):r.sha(p/x)for x in (*CACHE_FILES,'PLAN.json','CACHE.json')};allowed=dict(pins,**required)
    for key,value in list(r.seen.items()):require(r.sha(key)==value and allowed.get(key)==value,'Unbound consumed artifact '+key)
    actual=a['formation_rank'];reference=expected['arrays']['formation_rank'];valid=np.isfinite(reference)
    return dict(schema_version=1,passed=True,stage='prepare',plan_sha256=plan_sha,cache_sha256=r.sha(p/'CACHE.json'),
        orders_sha256=r.sha(p/'PRETRAIN_ORDERS.json'),**{k:True for k in CHECKS},source_hashes=sources,artifact_hashes=dict(r.seen),
        producer_modules_imported=False,returns_or_nav_computed=False,monthly_label_models_fit=0,new_model_fits=0,
        orders_checked=366,months_checked=122,candidate_cells_checked=5978,prefixes_checked=123,
        max_feature_error=float(np.max(np.abs(actual[valid]-reference[valid])))if valid.any() else 0.,
        max_score_error=max(float(np.max(np.abs(saved_scores[k]-scores[k])))for k in NAMES),
        price_parity=parity,
        baseline_episode_once_new_arms_bypass_episode=True,raw_finite_mask_and_core_rank_parity_verified=True,
        limitations=['New arms jointly replace the old boundary action; only worst-versus-mean isolates aggregation choice.',
            'No strategy returns computed. Inherited proxy-universe, price-vintage, and reused-history limits remain.'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--plan',type=Path,default=ROOT/STUDY/'PLAN.json');parser.add_argument('--prepare-complete',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/STUDY/'INDEPENDENT_PREPARE.json');args=parser.parse_args()
    require(args.prepare_complete,'Run only after registered preparation')
    result=audit(ROOT,args.plan,args.plan_sha)
    with args.output.open('x')as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(passed=True,output=str(args.output),sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__=='__main__':main()
