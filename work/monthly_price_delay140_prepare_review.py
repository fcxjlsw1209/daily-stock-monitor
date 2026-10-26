"""Independent round140 old-window price-delay reconstruction.

Actual market cache access occurs only in explicitly invoked audit(). Historical
regressions use QR/solve, independently of the producer's least-squares solver.
"""
from pathlib import Path
from datetime import timedelta
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
STUDY='outputs/monthly_price_delay140'
LOCAL=old.LOCAL
PROPOSAL_SHA='dafaf048f8139b3b1a486b37873115d5b551cb8741ce325b25aae9f87089319c'
CALENDAR_SHA='4b8f32ad8a555d6537d7cf3bd0f831e64fbb2de26d033c8daaf68590fcd4a6ea'
NAMES=('incumbent_local67','contemporaneous_nonfit','lagged_price_delay')
INPUTS=('proposal','meta','policy','original_orders','prices','sessions','protected_manifest')
CHECKS=('original27_arrays_byte_identical','all_features_independently_reconstructed',
        'all_three_score_matrices_and_ordered_choices_independently_verified',
        'one_episode_and_prefix_isolation_verified','zero_and_missing_exact_fallback')
EXTRA=('raw_nonfit','raw_delay','rank_nonfit','rank_delay','common_available')
CACHE_FILES=('DATA.npz','SCORES.npz','META.json','FEATURE_AUDIT.json','PRETRAIN_ORDERS.json','PRICE_PARITY.json')
ATOL=1e-12
require=old.require


def source_files(root=None):
    root=ROOT if root is None else Path(root).resolve()
    return sorted(set([root/'work/monthly_price_delay140_prepare_review.py',
                       root/'work/test_monthly_price_delay140_prepare_review.py',
                       root/'outputs/monthly_path70_research/incumbent_adapter.py']+old.source_files(root)))


def canonical_axis(values):
    axis=pd.DatetimeIndex(values)
    require(not axis.hasnans and axis.tz is None and axis.is_unique and axis.is_monotonic_increasing
            and axis.equals(axis.normalize()),'Ordered canonical daily axes')
    return axis


def endpoint_oracle(sessions,signal):
    axis=canonical_axis(sessions);day=pd.Timestamp(signal)
    require(day in axis,'Official signal date');position=axis.get_loc(day)
    if position<42:return None,pd.DatetimeIndex([]),pd.DatetimeIndex([])
    cutoff=axis[position-42];weeks={}
    for d in axis[:position-41]:
        friday=d+timedelta(days=(4-d.weekday())%7)
        if friday<=cutoff:weeks[friday]=d
    nominal=sorted(weeks)[-57:]
    return cutoff,pd.DatetimeIndex(nominal),pd.DatetimeIndex([weeks[x] for x in nominal])


def vector(values,n):
    a=np.asarray(values)
    require(a.shape==(n,) and all(isinstance(v,(int,float,np.integer,np.floating)) and not isinstance(v,(bool,np.bool_)) for v in a),'Numeric nonboolean vector')
    return np.array(a,dtype=float)


def empty_detail(reason):
    return dict(available=False,regression_available=False,reason=reason,restricted_r2=None,full_r2=None,
        restricted_condition=None,full_condition=None,restricted_rank=None,full_rank=None,
        response_variance=None,raw_nonfit=None,raw_delay=None)


def qr_oracle(stock_prices,market_prices):
    stock=vector(stock_prices,53);market=vector(market_prices,57)
    require((np.isfinite(stock)&(stock>0)).all() and (np.isfinite(market)&(market>0)).all(),'Positive finite endpoint prices')
    # Explicit row construction, independent of producer slicing.
    with np.errstate(all='raise'):
        y=np.array([stock[k+1]/stock[k]-1. for k in range(52)])
        market_returns=[market[k+1]/market[k]-1. for k in range(56)]
        regressors=np.array([[market_returns[t+4-lag] for lag in range(5)] for t in range(52)])
        centered=y-y.mean();sst=float(np.dot(centered,centered))
        require(math.isfinite(sst),'Finite response variation')
        out=empty_detail('zero_response_variance');out['response_variance']=sst/52
        if sst<=0:return out
        designs=[]
        for width in (1,5):
            columns=[]
            for j in range(width):
                values=regressors[:,j];deviation=values-values.mean();scale=float(np.sqrt(np.mean(deviation*deviation)))
                if not math.isfinite(scale) or scale<=0:out['reason']='zero_market_variance';return out
                columns.append(deviation/scale)
            designs.append(np.column_stack([np.ones(52)]+columns))
        fits={}
        for name,design in zip(('restricted','full'),designs):
            singular=np.linalg.svd(design,compute_uv=False)
            rank=int(np.count_nonzero(singular>singular[0]*1e-12))
            condition=float(singular[0]/singular[-1]) if singular[-1]>0 else None
            out[name+'_rank']=rank;out[name+'_condition']=condition
            if rank!=design.shape[1]:out['reason']='rank_deficient_'+name;return out
            if condition is None or not math.isfinite(condition) or condition>1e8:out['reason']='ill_conditioned_'+name;return out
            q,r=np.linalg.qr(design,mode='reduced');coefficients=np.linalg.solve(r,q.T@y)
            residual=y-design@coefficients;r2=1.-float(np.dot(residual,residual))/sst
            require(math.isfinite(r2) and -1e-10<=r2<=1.+1e-10,'R2 numerical bounds')
            fits[name]=max(0.,min(1.,r2))
        require(fits['restricted']<=fits['full']+1e-10,'Nested R2 order')
        fits['restricted']=min(fits['restricted'],fits['full'])
        out.update(restricted_r2=fits['restricted'],full_r2=fits['full'])
        if fits['full']<1e-8:out['reason']='full_r2_below_numeric_floor';return out
        out.update(available=True,regression_available=True,reason='available',raw_nonfit=1.-fits['restricted'],raw_delay=1.-fits['restricted']/fits['full'])
        return out


def rank_oracle(values):
    out=np.full(len(values),np.nan);ids=[i for i,v in enumerate(values) if math.isfinite(v)]
    if len(ids)<2:return out
    for i in ids:
        below=sum(values[j]<values[i] for j in ids);equal=sum(values[j]==values[i] for j in ids)
        out[i]=(below+(equal+1)/2.-1.)/(len(ids)-1)-.5
    return out


def cells_oracle(panel,meta,sessions):
    prices=panel['adj'];calendar=canonical_axis(sessions);dates=canonical_axis(meta['dates']);axis=canonical_axis(prices.index)
    require(isinstance(prices,pd.DataFrame) and prices.columns.is_unique,'Unique price frame')
    require(len(calendar)>0 and len(dates)==len(meta['symbols']),'Axes')
    require(len(meta['etfs'])==len(set(meta['etfs']))==17,'Original17 ETFs')
    if len(dates):require(axis[(axis>=calendar[0]) & (axis<=dates[-1])].isin(calendar).all(),'Official source price dates')
    cells=[]
    for i,(day,names) in enumerate(zip(dates,meta['symbols'])):
        require(len(names)==len(set(names))==49 and set(meta['etfs'])<=set(names),'Original49 pool')
        cutoff,nominal,actual=endpoint_oracle(calendar,day)
        x=None if len(actual)!=57 else vector(prices.reindex(index=actual,columns=['SPY']).to_numpy()[:,0],57)
        valid_market=x is not None and (np.isfinite(x)&(x>0)).all()
        for j,symbol in enumerate(names):
            detail=empty_detail('etf_not_applicable')
            if symbol not in meta['etfs']:
                if len(actual)!=57:detail=empty_detail('insufficient_history')
                elif not valid_market:detail=empty_detail('invalid_market_path')
                else:
                    y=vector(prices.reindex(index=actual[-53:],columns=[symbol]).to_numpy()[:,0],53)
                    detail=qr_oracle(y,x) if (np.isfinite(y)&(y>0)).all() else empty_detail('invalid_stock_path')
            cells.append(dict(signal_index=i,pool_index=j,signal_date=str(day.date()),symbol=symbol,
                benchmark=None if symbol in meta['etfs'] else 'SPY',cutoff=None if cutoff is None else str(cutoff.date()),
                nominal_week_ends=[str(v.date()) for v in nominal],actual_week_ends=[str(v.date()) for v in actual],
                market_window_start=str(actual[0].date()) if len(actual) else None,
                stock_window_start=str(actual[-53].date()) if len(actual)>=53 else None,
                window_end=str(actual[-1].date()) if len(actual) else None,**detail,rank_nonfit=None,rank_delay=None))
    return cells


def features_oracle(panel,meta,sessions,validated_raw=None):
    cells=cells_oracle(panel,meta,sessions);shape=(len(meta['dates']),49)
    arrays={key:np.full(shape,np.nan) for key in EXTRA[:-1]};arrays['common_available']=np.zeros(shape,bool)
    for i in range(shape[0]):
        batch=cells[i*49:(i+1)*49];valid=[j for j,c in enumerate(batch) if c['regression_available']]
        if len(valid)<2:
            for j in valid:batch[j].update(available=False,reason='insufficient_cross_section')
            continue
        for j in valid:
            arrays['common_available'][i,j]=True
            for key in EXTRA[:2]:arrays[key][i,j]=batch[j][key]
    if validated_raw is not None:
        for key in EXTRA[:2]:
            ref=validated_raw[key];require(ref.shape==shape and ref.dtype==np.dtype(float),'Raw feature schema')
            require(np.array_equal(np.isfinite(ref),arrays['common_available']) and not np.isinf(ref).any(),'Exact raw availability')
            np.testing.assert_allclose(ref,arrays[key],atol=ATOL,rtol=0,equal_nan=True)
    for i in range(shape[0]):
        for raw,rank in zip(EXTRA[:2],EXTRA[2:4]):
            values=arrays[raw][i] if validated_raw is None else validated_raw[raw][i]
            arrays[rank][i]=rank_oracle(values)
            for j,value in enumerate(arrays[rank][i]):
                if math.isfinite(value):cells[i*49+j][rank]=float(value)
    return dict(arrays=arrays,audit=dict(original_arrays_unchanged=True,original_months=shape[0],original_width=49,
        new_model_fits=0,monthly_label_models_fit=0,returns_or_nav_computed=False,feature_regressions_are_descriptive=True,
        new_model_fits_semantics='new predictive label/model fits only',
        attempted_stock_windows=sum(c['response_variance'] is not None for c in cells),
        descriptive_ols_calls=sum(int(c['restricted_rank'] is not None)+int(c['full_rank'] is not None) for c in cells),
        completed_regression_pairs=sum(c['full_rank'] is not None for c in cells),
        valid_regression_pairs=sum(c['regression_available'] for c in cells),
        original_auxiliary_states_and_core_coefficients_preserved=True,cells=cells))


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
    require(plan.get('round')==140 and plan.get('status')=='REGISTERED' and plan.get('arms')==list(NAMES)
            and plan.get('primary_challenger')==NAMES[2],'Round140 policy')
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
    require(len(original)==27 and set(a)==set(original)|set(EXTRA),'32 array schema')
    for key,value in original.items():
        require(a[key].shape==value.shape and a[key].dtype==value.dtype and a[key].tobytes()==value.tobytes(),'Original array bytes '+key)
    require(r.sha(old.LOADER)==old.LOADER_SHA,'Raw loader code')
    spec=importlib.util.spec_from_file_location('delay140_raw_loader',r.path(old.LOADER))
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
    expected=features_oracle(panel,meta,sessions,validated_raw=a)
    compare(r.json(p/'FEATURE_AUDIT.json'),expected['audit'],'FEATURE_AUDIT')
    for key in EXTRA:
        require(a[key].shape==(122,49) and a[key].dtype==expected['arrays'][key].dtype,'Feature schema '+key)
        if key in EXTRA[:2]:np.testing.assert_allclose(a[key],expected['arrays'][key],rtol=0,atol=ATOL,equal_nan=True)
        else:np.testing.assert_array_equal(a[key],expected['arrays'][key])
    beta=expected['arrays']['rank_nonfit'];rho=expected['arrays']['rank_delay']
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
    for n in (0,1,36,53,60,76,88,122):
        cut=panel['adj'].loc[:meta['dates'][n-1]] if n else panel['adj'].iloc[:0]
        subset=dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n])
        prefix=features_oracle(dict(panel,adj=cut),subset,sessions,validated_raw={k:a[k][:n] for k in EXTRA[:2]})
        compare(prefix['audit']['cells'],expected['audit']['cells'][:n*49],'Source prefix')
        for key in EXTRA:np.testing.assert_array_equal(prefix['arrays'][key],expected['arrays'][key][:n])
        ss,cc,_=score_oracle(a,meta,prefix['arrays']['rank_nonfit'],prefix['arrays']['rank_delay'],n)
        for name in NAMES:require(cc[name]==choices[name][:n] and ss[name].tobytes()==scores[name][:n].tobytes(),'Score/choices prefix')
    for fill in (np.nan,0.):
        hand=np.full((122,49),np.nan)
        for i,names in enumerate(meta['symbols']):
            for j,symbol in enumerate(names):
                if symbol not in meta['etfs']:hand[i,j]=fill
        ss,cc,_=score_oracle(a,meta,hand,hand)
        for name in NAMES[1:]:require(ss[name].tobytes()==scores[NAMES[0]].tobytes() and cc[name]==choices[NAMES[0]],'Missing/rankedzero exact fallback')
    required={str(p/x):r.sha(p/x) for x in (*CACHE_FILES,'PLAN.json','CACHE.json')};allowed=dict(pins,**required)
    for key,value in list(r.seen.items()):require(r.sha(key)==value and allowed.get(key)==value,'Unbound consumed artifact '+key)
    valid=expected['arrays']['common_available']
    errors={key:float(np.max(np.abs(a[key][valid]-expected['arrays'][key][valid]))) if valid.any() else 0. for key in EXTRA[:2]}
    return dict(schema_version=1,passed=True,stage='prepare',plan_sha256=plan_sha,cache_sha256=r.sha(p/'CACHE.json'),
        orders_sha256=r.sha(p/'PRETRAIN_ORDERS.json'),**{k:True for k in CHECKS},source_hashes=sources,artifact_hashes=dict(r.seen),
        producer_modules_imported=False,returns_or_nav_computed=False,monthly_label_models_fit=0,new_model_fits=0,
        orders_checked=366,source_cells_checked=5978,stock_windows_checked=3904,available_stock_cells=int(valid.sum()),
        max_feature_errors=errors,max_score_error=max(float(np.max(np.abs(saved_scores[k]-scores[k]))) for k in NAMES),
        raw_prices_independently_verified=True,calendar_overlay_independently_verified=True,price_parity=parity,
        weekly_endpoints_independently_verified=True,all_signal_prefixes_verified=True,descriptive_ols_allowed=True,
        raw_zero_uses_ordinary_average_rank=True,ranked_zero_and_missing_exact_baseline=True,
        limitations=['Independent QR verifies historical OLS fits; exact ranks use validated producer binary64 raw values. Historical descriptive regressions are not new predictive model fits.',
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
