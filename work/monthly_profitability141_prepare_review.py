"""Independent round141 annual profitability reconstruction.

Actual market cache access occurs only in explicitly invoked audit(). Exact Fraction division is independent of the producer's Decimal cross-products.
"""
from pathlib import Path
from datetime import date, timedelta
from decimal import Decimal
from fractions import Fraction
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
import monthly_latent_state134_prepare_review as old

ROOT=Path(__file__).resolve().parents[1]
STUDY='outputs/monthly_profitability141'
LOCAL=old.LOCAL
PROPOSAL_SHA='80901aab35840451f289e58ca90caefefa751c717561e8cb1175d39012fd7a58'
CALENDAR_SHA='4b8f32ad8a555d6537d7cf3bd0f831e64fbb2de26d033c8daaf68590fcd4a6ea'
NAMES=('incumbent_local67','annual_gross_profit_growth_sign','annual_gross_profitability_improvement_sign')
INPUTS=('proposal','meta','policy','original_orders','prices','sessions','protected_manifest','admitted_dataset','source_admission')
CHECKS=('original27_arrays_byte_identical','all_features_independently_reconstructed',
        'all_three_score_matrices_and_ordered_choices_independently_verified',
        'one_episode_and_prefix_isolation_verified','zero_and_missing_exact_fallback')
EXTRA=('gross_profit_growth_sign','gross_profitability_improvement_sign','common_available')
CACHE_FILES=('DATA.npz','SCORES.npz','META.json','FEATURE_AUDIT.json','PRETRAIN_ORDERS.json','PRICE_PARITY.json')
ATOL=1e-12
require=old.require


def source_files(root=None):
    root=ROOT if root is None else Path(root).resolve()
    return sorted(set([root/'work/monthly_profitability141_prepare_review.py',
                       root/'work/test_monthly_profitability141_prepare_review.py',
                       root/'outputs/monthly_path70_research/incumbent_adapter.py']+old.source_files(root)))


def canonical_axis(values):
    axis=pd.DatetimeIndex(values)
    require(not axis.hasnans and axis.tz is None and axis.is_unique and axis.is_monotonic_increasing
            and axis.equals(axis.normalize()),'Ordered canonical daily axes')
    return axis


def parse_day(value):
    require(type(value) is str, 'Explicit ISO date')
    out=date.fromisoformat(value)
    require(out.isoformat()==value,'Canonical ISO date')
    return out


def exact_amount(value):
    require(isinstance(value,dict),'Value-scale pair')
    entries=[]
    for name in ('value','scale'):
        raw=value[name]
        require(type(raw) is str and raw and raw.strip()==raw,'Explicit decimal string')
        number=Decimal(raw)
        require(number.is_finite(),'Finite amount')
        entries.append(Fraction(number))
    require(entries[1]>0,'Positive scale')
    return entries[0]*entries[1]


def valid_event(event,identity_day):
    try:
        require(all(event.get(k) is True for k in ('admitted','original_annual','measurement_eligible')),'Admitted original eligible event')
        require(all(type(event.get(k)) is str and event[k] and event[k].strip()==event[k]
                    for k in ('issuer_id','security_id','currency_key')),'Identity and currency')
        start,end,available=[parse_day(event[k]) for k in ('period_start','period_end','available_from')]
        require(parse_day(event['identity_valid_from'])<=identity_day<=parse_day(event['identity_valid_through']),'Identity date')
        length=(end-start).days+1
        require(330<=length<=400 and end<available,'Annual chronology')
        gross,assets=exact_amount(event['gross_profit']),exact_amount(event['assets'])
        require(assets>0,'Positive assets')
        return dict(start=start,end=end,available=available,days=length,gross=gross,assets=assets)
    except (KeyError,TypeError,ValueError,ArithmeticError):return None


def newest(events):
    if not events:return None
    ordered=sorted(events,key=lambda x:x['available_from'],reverse=True)
    if len(ordered)>1 and ordered[0]['available_from']==ordered[1]['available_from']:return None
    return ordered[0]


def cell_oracle(events,signal):
    day=parse_day(signal);seen={}
    out=dict(available=False,current_event_id=None,prior_event_id=None,
             current_available_from=None,prior_available_from=None,
             current_period_end=None,prior_period_end=None,
             current_days=None,prior_days=None,currency_key=None,
             gross_profit_growth_sign=None,gross_profitability_improvement_sign=None)
    require(type(events) is list,'Event list')
    for event in events:
        require(isinstance(event,dict) and type(event.get('event_id')) is str and event['event_id'].strip()==event['event_id'] and bool(event['event_id']),'Event identifier')
        if parse_day(event['available_from'])>day:continue
        if event['event_id'] in seen and seen[event['event_id']]!=event:return out
        seen[event['event_id']]=event
    known=list(seen.values());current=newest(known)
    if current is None:return out
    out.update(current_event_id=current['event_id'],current_available_from=current['available_from']);cur=valid_event(current,day)
    if cur is None or (day-cur['end']).days>550:return out
    prior=newest([e for e in known if e['event_id']!=current['event_id'] and e.get('period_end')==(cur['start']-timedelta(days=1)).isoformat()])
    if prior is None:return out
    out.update(prior_event_id=prior['event_id'],prior_available_from=prior['available_from'])
    for event in known:
        if event['event_id'] not in (current['event_id'],prior['event_id']) and event['available_from']>=prior['available_from']:
            try:parse_day(event.get('period_end'))
            except (TypeError,ValueError):return out
    prev=valid_event(prior,parse_day(prior['available_from']))
    if prev is None:return out
    out.update(current_period_end=cur['end'].isoformat(),prior_period_end=prev['end'].isoformat(),
               current_days=cur['days'],prior_days=prev['days'],currency_key=current['currency_key'])
    if any(current[k]!=prior[k] for k in ('issuer_id','security_id','currency_key')):return out
    if prev['end']+timedelta(days=1)!=cur['start'] or not 330<=(cur['end']-prev['end']).days<=400:return out
    g0,g1=prev['gross']/prev['days'],cur['gross']/cur['days']
    p0,p1=g0/prev['assets'],g1/cur['assets']
    out.update(available=True,gross_profit_growth_sign=float((g1>g0)-(g1<g0))/2,
               gross_profitability_improvement_sign=float((p1>p0)-(p1<p0))/2)
    return out


def features_oracle(meta,dataset):
    require(type(dataset.get('schema_version')) is int and dataset['schema_version']==1 and isinstance(dataset.get('events'),dict),'Dataset schema')
    dates=canonical_axis(meta['dates']);require(len(dates)==len(meta['symbols']),'Signal axes')
    etfs=meta['etfs'];require(len(etfs)==len(set(etfs))==17,'Original17 ETFs')
    shape=(len(dates),49);a={k:np.full(shape,np.nan) for k in EXTRA[:-1]};a[EXTRA[-1]]=np.zeros(shape,bool);cells=[]
    for i,(day,names) in enumerate(zip(dates,meta['symbols'])):
        require(len(names)==len(set(names))==49 and set(etfs)<=set(names),'Original49 candidates')
        for j,symbol in enumerate(names):
            value=cell_oracle([] if symbol in etfs else dataset['events'].get(symbol,[]),str(day.date()))
            value.update(signal_index=i,pool_index=j,signal_date=str(day.date()),symbol=symbol)
            cells.append(value)
            if value['available']:
                a[EXTRA[-1]][i,j]=True
                for key in EXTRA[:-1]:a[key][i,j]=value[key]
    return dict(arrays=a,cells=cells)


def verify_audit(saved,expected,*,dataset_sha=None,admission_sha=None):
    require(type(saved.get('new_model_fits')) is int and saved['new_model_fits']==0 and saved.get('returns_or_nav_computed') is False,'No new fits or returns')
    if dataset_sha is not None or admission_sha is not None:
        require(bool(dataset_sha) and bool(admission_sha),'Both source audit bindings required')
        require(saved.get('admitted_dataset_sha256')==dataset_sha and
                saved.get('source_admission_sha256')==admission_sha,'Feature audit source bindings')
    require(saved.get('original_arrays_unchanged') is True and saved.get('original_auxiliary_states_and_core_coefficients_preserved') is True,'Original state preserved')
    require(saved.get('latest_event_before_validity_filter') is True,'Chronology before validity')
    require(saved.get('common_available_cells')==int(expected['arrays']['common_available'].sum()),'Common count')
    require(saved.get('original_months')==len(expected['cells'])//49 and saved.get('original_width')==49,'Audit axes')
    require(len(saved.get('cells',[]))==len(expected['cells']),'Every cell audited')
    for observed,wanted in zip(saved['cells'],expected['cells']):
        for key,value in wanted.items():
            require(type(observed.get(key)) is type(value) and observed[key]==value,'Independent cell mismatch: '+key)
        if wanted['available']:require(observed.get('reason')=='available','Valid reason')


def score_oracle(a,meta,nonfit_hand,delay_hand,n=None):
    h=old.original_helper();x=h.original_prefix(a,meta,n);n=x['n'];shape=x['core'].shape
    hands=[np.asarray(hand)[:n]for hand in (nonfit_hand,delay_hand)]
    for hand in hands:
        require(hand.shape==shape and hand.dtype.kind in 'fiu'and not np.isinf(hand).any(),'Hand shape/type')
        require((np.abs(hand[np.isfinite(hand)])<=.5).all(),'Bounded hand')
        for i,names in enumerate(meta['symbols'][:n]):
            require(all(not np.isfinite(hand[i,j])for j,s in enumerate(names)if s in meta['etfs']),'No ETF feature')
    require(np.array_equal(np.isfinite(hands[0]),np.isfinite(hands[1])),'Matched hand mask')
    scores={};choices={};deltas={}
    for k,name in enumerate(NAMES):
        pre=x['local'].copy();delta=np.zeros(shape)
        if k:
            hand=hands[k-1]
            for i in range(n):
                for j in range(shape[1]):
                    if not x['eligible'][i,j]or not np.isfinite(hand[i,j])or hand[i,j]==0.:continue
                    s=x['support'][i,j];delta[i,j]=.025*(min(.5,max(-.5,s+float(hand[i,j])))-s)
            require((np.abs(delta)<=.0125+1e-12).all(),'Small correction bound')
            pre[delta!=0.]+=delta[delta!=0.]
        c,s,_=h.episode(pre,x);scores[name]=s;choices[name]=c;deltas[name]=delta
    return scores,choices,deltas


def compare(actual,expected,path=''):
    if isinstance(expected,dict):
        require(isinstance(actual,dict)and set(actual)==set(expected),'Schema '+path)
        for key in expected:compare(actual[key],expected[key],path+'/'+key)
    elif isinstance(expected,list):
        require(isinstance(actual,list)and len(actual)==len(expected),'List '+path)
        for i,(a,b)in enumerate(zip(actual,expected)):compare(a,b,path+'/'+str(i))
    elif type(expected)is float:
        require(type(actual)in (float,int)and math.isfinite(actual)and math.isclose(actual,expected,abs_tol=ATOL,rel_tol=1e-12 if path.endswith("condition") else 0),'Numeric '+path)
    else:require(type(actual)is type(expected)and actual==expected,'Discrete '+path)


def audit(root,plan_path,plan_sha):
    require(re.fullmatch('[0-9a-f]{64}',plan_sha) is not None,'Explicit PLAN SHA')
    r=old.Reader(root);p=r.path(STUDY);require(r.path(plan_path)==p/'PLAN.json','Dedicated PLAN')
    require(r.sha(plan_path)==plan_sha,'PLAN SHA');plan=r.json(plan_path)
    require(plan.get('round')==141 and plan.get('status')=='REGISTERED' and plan.get('arms')==list(NAMES)
            and plan.get('primary_challenger')==NAMES[2],'Round141 policy')
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
    require(len(original)==27 and set(a)==set(original)|set(EXTRA),'30 array schema')
    for key,value in original.items():
        require(a[key].shape==value.shape and a[key].dtype==value.dtype and a[key].tobytes()==value.tobytes(),'Original array bytes '+key)
    require(r.sha(old.LOADER)==old.LOADER_SHA,'Raw loader code')
    spec=importlib.util.spec_from_file_location('profitability141_raw_loader',r.path(old.LOADER))
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
    admission=r.json(inputs['source_admission']);dataset=r.json(inputs['admitted_dataset'])
    require(admission.get('round')==141 and admission.get('stage')=='source_admission' and admission.get('passed') is True,'Independent source admission')
    require(all(admission.get(k) is True for k in ('semantic_review_passed','availability_review_passed','identity_review_passed','period_review_passed')),'Source review coverage')
    for key,expected_hash in [('dataset_sha256',r.sha(inputs['admitted_dataset'])),('meta_sha256',r.sha(inputs['meta'])),('sessions_sha256',r.sha(inputs['sessions'])),('proposal_sha256',PROPOSAL_SHA)]:
        require(admission.get(key)==expected_hash,'Admission binding '+key)
    require(type(admission.get('factor_or_return_calculations')) is int and admission['factor_or_return_calculations']==0,'Admission before calculation')
    facts=r.verify(admission['source_hashes']);require(facts==r.verify(plan['fundamental_source_hashes']) and all(pins.get(k)==v for k,v in facts.items()),'Fundamental source closure')
    expected=features_oracle(meta,dataset)
    verify_audit(r.json(p/'FEATURE_AUDIT.json'),expected,
                 dataset_sha=r.sha(inputs['admitted_dataset']),admission_sha=r.sha(inputs['source_admission']))
    for key in EXTRA:
        require(a[key].shape==(122,49) and a[key].dtype==expected['arrays'][key].dtype,'Feature schema '+key)
        np.testing.assert_array_equal(a[key],expected['arrays'][key])
    beta=expected['arrays'][EXTRA[0]];rho=expected['arrays'][EXTRA[1]]
    scores,choices,deltas=score_oracle(a,meta,beta,rho)
    saved_scores=r.npz(p/'SCORES.npz');saved_orders=r.json(p/'PRETRAIN_ORDERS.json')
    require(set(saved_scores)==set(saved_orders)==set(NAMES),'Three output arms')
    for name in NAMES:
        wanted=[dict(date=day,signal_date=day,end=meta['ends'][i],symbols=[meta['symbols'][i][j] for j in choice],
            scores=[float(scores[name][i,j]) for j in choice]) for i,(day,choice) in enumerate(zip(meta['dates'],choices[name]))]
        compare(saved_orders[name],wanted,'orders/'+name)
        np.testing.assert_allclose(saved_scores[name],scores[name],rtol=0,atol=ATOL)
    h.assert_exact_tree(saved_orders[NAMES[0]],r.json(inputs['original_orders']))
    require(saved_scores[NAMES[0]].tobytes()==scores[NAMES[0]].tobytes(),'Baseline score bytes')
    for n in range(len(meta['dates'])+1):
        subset=dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n])
        historical=dict(dataset,events={symbol:[e for e in events if n and e['available_from']<=meta['dates'][n-1]] for symbol,events in dataset['events'].items()})
        prefix=features_oracle(subset,historical)
        compare(prefix['cells'],expected['cells'][:n*49],'Source prefix')
        for key in EXTRA:np.testing.assert_array_equal(prefix['arrays'][key],expected['arrays'][key][:n])
        ss,cc,_=score_oracle(a,meta,prefix['arrays'][EXTRA[0]],prefix['arrays'][EXTRA[1]],n)
        for name in NAMES:require(cc[name]==choices[name][:n] and ss[name].tobytes()==scores[name][:n].tobytes(),'Score/choices prefix')
    for fill in (np.nan,0.):
        hand=np.full((122,49),np.nan)
        for i,names in enumerate(meta['symbols']):
            for j,symbol in enumerate(names):
                if symbol not in meta['etfs']:hand[i,j]=fill
        ss,cc,_=score_oracle(a,meta,hand,hand)
        for name in NAMES[1:]:require(ss[name].tobytes()==scores[NAMES[0]].tobytes() and cc[name]==choices[NAMES[0]],'Missing/zero exact fallback')
    required={str(p/x):r.sha(p/x) for x in (*CACHE_FILES,'PLAN.json','CACHE.json')};allowed=dict(pins,**required)
    for key,value in list(r.seen.items()):require(r.sha(key)==value and allowed.get(key)==value,'Unbound consumed artifact '+key)
    valid=expected['arrays']['common_available']
    errors={key:float(np.max(np.abs(a[key][valid]-expected['arrays'][key][valid]))) if valid.any() else 0. for key in EXTRA[:2]}
    return dict(schema_version=1,passed=True,stage='prepare',plan_sha256=plan_sha,cache_sha256=r.sha(p/'CACHE.json'),
        orders_sha256=r.sha(p/'PRETRAIN_ORDERS.json'),**{k:True for k in CHECKS},source_hashes=sources,artifact_hashes=dict(r.seen),
        producer_modules_imported=False,returns_or_nav_computed=False,monthly_label_models_fit=0,new_model_fits=0,
        orders_checked=366,source_cells_checked=5978,stock_cells_checked=3904,available_stock_cells=int(valid.sum()),
        max_feature_errors=errors,max_score_error=max(float(np.max(np.abs(saved_scores[k]-scores[k]))) for k in NAMES),
        raw_prices_independently_verified=True,calendar_overlay_independently_verified=True,price_parity=parity,
        source_admission_bound=True,common_availability_verified=True,admitted_report_pairs_independently_reconstructed=True,all_signal_prefixes_verified=True,exact_fraction_signs=True,
        exact_zero_and_missing_baseline=True,
        limitations=['Independent Fraction division verifies Decimal cross-product signs; limited MSFT/NVDA current-retrieval issuer reports are not certified first-public versions.',
            'No monthly prediction models or strategy returns computed; inherited proxy-universe and price-vintage limitations remain.'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--plan',type=Path,default=ROOT/STUDY/'PLAN.json');parser.add_argument('--prepare-complete',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/STUDY/'INDEPENDENT_PREPARE.json');args=parser.parse_args()
    require(args.prepare_complete,'Run only after registered preparation')
    result=audit(ROOT,args.plan,args.plan_sha)
    with args.output.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(passed=True,output=str(args.output),sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__=='__main__':main()
