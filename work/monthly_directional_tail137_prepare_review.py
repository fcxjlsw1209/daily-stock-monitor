"""Independent round137 source/state/core/ordering review; synthetic-safe import.

The directional state is reconstructed with Decimal. Frozen independent raw
price, calendar and original-core routines are reused without HMM fitting.
"""
from pathlib import Path
from decimal import Decimal, localcontext
import argparse
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
STUDY='outputs/monthly_directional_tail137'
LOCAL=old.LOCAL
NAMES=('incumbent_local67','constant_half_core','directional_tail_core')
EXTRA=('directional_q','directional_raw_rank','directional_risk_rank')
INPUTS=('proposal','meta','policy','original_orders','prices','sessions','transition','protected_manifest')
CACHE_FILES=old.CACHE_FILES
CHECKS=('original27_arrays_byte_identical','all_features_independently_reconstructed',
        'all_three_score_matrices_and_ordered_choices_independently_verified',
        'one_episode_and_prefix_isolation_verified','zero_and_missing_exact_fallback')
PROPOSAL_SHA='1227a695b36aa62c91bcbb4b19e421f40c11cc75a5afb8377a5946b6fd8b709b'
CALENDAR_SHA='4b8f32ad8a555d6537d7cf3bd0f831e64fbb2de26d033c8daaf68590fcd4a6ea'
STATE_ATOL=1e-12
SCORE_ATOL=1e-12
require=old.require


def source_files(root=None):
    root=ROOT if root is None else Path(root).resolve()
    return sorted(set([root/'work/monthly_directional_tail137_prepare_review.py',
                       root/'work/test_monthly_directional_tail137_prepare_review.py']+old.source_files(root)))


def empty_detail(reason):
    return dict(available=False,reason=reason,stock_returns=None,mean_return=None,centered_scale=None,
                positive_sum_squares_scaled=None,negative_sum_squares_scaled=None,
                positive_count=None,negative_count=None,zero_count=None,q=None)


def tail_oracle(prices):
    arr=np.asarray(prices)
    require(arr.shape==(22,32),'Exactly22 official rows and32 stock columns')
    if arr.dtype.kind not in 'fiu':
        require(arr.dtype.kind=='O' and all(isinstance(v,(int,float,np.integer,np.floating))
                and not isinstance(v,(bool,np.bool_)) for v in arr.flat),'Numeric prices, not booleans or text')
    arr=np.asarray(arr,dtype=float)
    # Completeness precedes ratio computation for every stock.
    if not np.isfinite(arr).all() or not (arr>0).all():return float('nan'),empty_detail('invalid_price')
    returns=[]
    with localcontext() as context:
        context.prec=2500
        for j in range(32):
            # Independent high-precision division rounded once to the source
            # binary64 ratio, followed by the specified simple-return subtraction.
            ratio=Decimal.from_float(float(arr[-1,j]))/Decimal.from_float(float(arr[0,j]))
            value=float(ratio)-1.
            require(math.isfinite(value),'Unrepresentable simple return')
            returns.append(value)
        xs=[Decimal.from_float(x) for x in returns]
        mean=sum(xs)/32;centered=[x-mean for x in xs];scale=max(abs(x)for x in centered)
        if not scale:
            detail=empty_detail('zero_dispersion')
            detail.update(stock_returns=returns,mean_return=float(mean),centered_scale=0.,
                          positive_count=0,negative_count=0,zero_count=32,
                          positive_sum_squares_scaled=0.,negative_sum_squares_scaled=0.)
            return float('nan'),detail
        scaled=[x/scale for x in centered]
        upper=sum(x*x for x in scaled if x>0);lower=sum(x*x for x in scaled if x<0)
        q=float(upper/(upper+lower))
        detail=dict(available=True,reason='available',stock_returns=returns,mean_return=float(mean),
                    centered_scale=float(scale),positive_sum_squares_scaled=float(upper),
                    negative_sum_squares_scaled=float(lower),positive_count=sum(x>0 for x in centered),
                    negative_count=sum(x<0 for x in centered),zero_count=sum(x==0 for x in centered),q=q)
        require(0<=q<=1,'Bounded relative semivariance')
        return q,detail


def state_oracle(prices,meta,sessions):
    require(isinstance(prices,pd.DataFrame),'Price frame')
    idx=pd.DatetimeIndex(prices.index);calendar=pd.DatetimeIndex(sessions)
    require(idx.tz is None and not idx.hasnans and idx.is_unique and idx.is_monotonic_increasing
            and idx.equals(idx.normalize()) and prices.columns.is_unique,'Price axes')
    require(calendar.tz is None and not calendar.hasnans and calendar.is_unique and calendar.is_monotonic_increasing
            and calendar.equals(calendar.normalize()),'Official session axis')
    require(len(meta['dates'])==len(meta['symbols']),'Signal axes')
    etfs=meta['etfs'];require(len(etfs)==len(set(etfs))==17,'Original17 ETF identifiers')
    dates=pd.DatetimeIndex(meta['dates'])
    require(not dates.hasnans and dates.tz is None and dates.equals(dates.normalize()),'Signal dates')
    if len(dates):
        historical=idx[(idx>=calendar[0])&(idx<=dates[-1])]
        require(historical.isin(calendar).all(),'Nonofficial historical price row')
    values=[];months=[];last=None
    for i,(date,names)in enumerate(zip(meta['dates'],meta['symbols'])):
        day=pd.Timestamp(date);require(day in calendar and (last is None or day>last),'Ordered official signal')
        require(len(names)==len(set(names))==49 and set(etfs)<=set(names),'Original49-name pool')
        stocks=[x for x in names if x not in etfs];require(len(stocks)==32,'Exactly32 stocks')
        at=calendar.get_loc(day);window=list(calendar[at-21:at+1])if at>=21 else []
        if len(window)!=22:q,detail=float('nan'),empty_detail('insufficient_history')
        elif not set(stocks)<=set(prices.columns):q,detail=float('nan'),empty_detail('missing_stock_columns')
        else:q,detail=tail_oracle(prices.reindex(index=window,columns=stocks).to_numpy())
        dd=[str(x.date())for x in window]
        detail.update(signal_index=i,signal_date=str(day.date()),stock_symbols=stocks,
                      window_dates=dd,window_start=dd[0]if dd else None,window_end=dd[-1]if dd else None)
        values.append(q);months.append(detail);last=day
    return dict(q=np.asarray(values,dtype=float),audit=dict(months=months,new_model_fits=0,return_sessions=21,
                price_rows=22,stock_count=32,etf_count=17,no_eligibility_filter=True))


def score_oracle(a,meta,raw,risk,q,n=None):
    h=old.original_helper();x=h.original_prefix(a,meta,n);n=x['n']
    raw=np.asarray(raw)[:n];risk=np.asarray(risk)[:n];values=np.asarray(q)[:n]
    require(raw.shape==risk.shape==x['core'].shape and values.shape==(n,),'State/core axes')
    require(values.dtype.kind in 'fiu' and not np.isinf(values).any(),'Finite or missing numeric state')
    scores={};choices={};deltas={}
    for name in NAMES:
        pre=x['local'].copy();delta=np.zeros_like(pre)
        if name!=NAMES[0]:
            for i in range(n):
                if np.isnan(values[i]):continue
                require(0<=values[i]<=1,'Bounded q')
                weight=.5 if name==NAMES[1]else float(values[i])
                for j in range(pre.shape[1]):
                    u,v=raw[i,j],risk[i,j]
                    require(math.isfinite(u)and math.isfinite(v)and -.5-1e-12<=u<=.5+1e-12
                            and -.5-1e-12<=v<=.5+1e-12,'Bounded full-pool ranks')
                    mixed=weight*u+(1-weight)*v
                    delta[i,j]=.5*(x['cw'][i,j]*mixed-x['core'][i,j])
            pre[delta!=0]+=delta[delta!=0]
        cc,ss,_=h.episode(pre,x);scores[name]=ss;choices[name]=cc;deltas[name]=delta
    return scores,choices,deltas


def compare(actual,expected,path=''):
    if isinstance(expected,dict):
        require(isinstance(actual,dict)and set(actual)==set(expected),'Schema '+path)
        for k in expected:compare(actual[k],expected[k],path+'/'+k)
    elif isinstance(expected,list):
        require(isinstance(actual,list)and len(actual)==len(expected),'List '+path)
        for i,(a,b)in enumerate(zip(actual,expected)):compare(a,b,path+'/'+str(i))
    elif type(expected)is float:
        require(type(actual)in(int,float)and math.isfinite(actual)and math.isclose(actual,expected,abs_tol=STATE_ATOL,rel_tol=0),'Numeric '+path)
    else:require(type(actual)is type(expected)and actual==expected,'Discrete '+path)


def audit(root,plan_path,plan_sha):
    require(re.fullmatch('[0-9a-f]{64}',plan_sha) is not None,'Explicit PLAN SHA')
    r=old.Reader(root);p=r.path(STUDY);require(r.path(plan_path)==p/'PLAN.json','Dedicated PLAN')
    require(r.sha(plan_path)==plan_sha,'PLAN SHA');plan=r.json(plan_path)
    require(plan.get('round')==137 and plan.get('status')=='REGISTERED' and plan.get('arms')==list(NAMES)
            and plan.get('primary_challenger')==NAMES[2],'Round137 policy')
    require(r.path(plan['output_directory'])==p,'Output directory')
    require(plan['runtime']==dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),'Runtime')
    require(not any((p/x).exists() for x in ('TRAIN_TRIALS.json','FROZEN.json','RESULTS.json')),'Before outcome stages')
    pins=r.verify(plan['hashes']);require(set(plan['inputs'])==set(INPUTS),'Named inputs')
    inputs={}
    for k,ref in plan['inputs'].items():
        path=r.path(ref['path']);require(r.sha(path)==ref['sha256']==pins.get(str(path)),'Bound input '+k);inputs[k]=path
    require(r.sha(inputs['proposal'])==PROPOSAL_SHA,'Final proposal')
    require(plan.get('actual_model_fits_before_registration') == plan.get('actual_feature_calculations_before_registration') == plan.get('actual_returns_before_registration') == 0, 'Prospective registration')
    require(plan['tolerances'].get('state_atol')==1e-12 and plan['tolerances'].get('state_rtol')==0 and plan['tolerances'].get('scores_atol')==1e-12 and plan['tolerances'].get('scores_rtol')==0, 'Frozen numeric tolerances')
    sources={str(f):r.sha(f) for f in source_files(root)}
    require(r.verify(plan['independent_prepare_source_hashes'])==sources and all(pins.get(k)==v for k,v in sources.items()),'Independent closure')
    oldplan=r.json(LOCAL+'/PLAN.json');raw=r.verify(plan['raw_source_hashes'])
    require(raw==r.verify(oldplan['raw_source_hashes']) and len(raw)==64 and all(pins.get(k)==v for k,v in raw.items()),'Raw64 source identity')
    oldcache=r.verify(r.json(LOCAL+'/CACHE_HASHES.json'),r.path(LOCAL))
    require(all(pins.get(k)==v for k,v in oldcache.items()),'All original cache artifacts bound')
    for k,file in [('meta','META.json'),('policy','INCUMBENT_POLICY.json'),('original_orders','local_linear_ORDERS.json')]:
        require(inputs[k]==r.path(LOCAL+'/'+file),'Original wiring '+k)
    require(inputs['prices']==r.path(old.PRICES) and inputs['transition']==r.path('outputs/monthly_target70_research/TRANSITION.npz'),'Original numeric input wiring')
    h=old.original_helper();policy=r.json(inputs['policy']);require(all(policy.get(k)==v for k,v in h.POLICY.items()),'Frozen episode policy')
    cache=r.json(p/'CACHE.json');require(set(cache)==set(CACHE_FILES),'Seven cached outputs');r.verify(cache,p)
    meta=r.json(p/'META.json');require(meta==r.json(inputs['meta']),'Exact metadata')
    require(len(meta['dates'])==len(meta['symbols'])==122 and meta['train_months']==88 and meta['early_months']==53
            and len(meta['etfs'])==17 and all(len(x)==49 for x in meta['symbols']),'Original122x49')
    original=r.npz(LOCAL+'/DATA.npz');a=r.npz(p/'DATA.npz')
    extras=set(EXTRA)
    require(len(original)==27 and set(a)==set(original)|extras,'30 array schema')
    for k,v in original.items():require(a[k].shape==v.shape and a[k].dtype==v.dtype and a[k].tobytes()==v.tobytes(),'Original array bytes '+k)
    require(r.sha(old.LOADER)==old.LOADER_SHA,'Raw loader code')
    spec=importlib.util.spec_from_file_location('directional137_raw_loader',r.path(old.LOADER));loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    panel,info=loader.load_panel(meta['etfs']);used=dict(info['source_hashes']);used[str(r.path(old.LOADER))]=old.LOADER_SHA
    require({str(r.path(k)):v for k,v in used.items()}==raw,'Actually consumed raw sources');r.verify(used)
    prices=r.npz(inputs['prices']);require(set(prices)=={'dates','symbols','close','opening'},'Price cache schema')
    parity=old.price_parity(panel,prices);old.compare(r.json(p/'PRICE_PARITY.json'),parity,'PRICE_PARITY')
    calendar_path=inputs['sessions'].parent/'CALENDAR_PLAN.json'
    require(r.sha(calendar_path)==CALENDAR_SHA,'Calendar plan hash')
    calendar_plan=r.json(calendar_path);calendar_bindings=r.verify(calendar_plan['bindings'])
    require(all(pins.get(k)==v for k,v in calendar_bindings.items()),'Calendar source closure')
    require(calendar_plan['start']=='2015-01-01' and calendar_plan['end']=='2027-12-31','Fixed calendar coverage')
    sessions=old.official_sessions(r,calendar_plan['end'])
    sessions=sessions[sessions>=calendar_plan['start']]
    saved_sessions=r.json(inputs['sessions'])
    require(saved_sessions['plan_sha256']==CALENDAR_SHA and saved_sessions['sessions']==[str(d.date()) for d in sessions], 'All official sessions and overlay')
    frame=pd.DataFrame(prices['close'],index=pd.to_datetime(prices['dates']),columns=prices['symbols'])
    raw_rank,risk_rank,cw,eligible=old.reconstruct_core(panel,meta);tr=r.npz(inputs['transition'])
    np.testing.assert_array_equal(eligible,original['eligible']);np.testing.assert_array_equal(eligible,tr['eligible'])
    np.testing.assert_array_equal(.5*raw_rank,tr['calm_core']);np.testing.assert_array_equal(.52*risk_rank,tr['noncalm_core'])
    np.testing.assert_array_equal(raw_rank,a['directional_raw_rank']);np.testing.assert_array_equal(risk_rank,a['directional_risk_rank'])
    expected_cw=np.broadcast_to(cw[:,None],(122,49));np.testing.assert_array_equal(expected_cw,tr['cw'])
    expected_core=np.where(cw[:,None]==.5,.5*raw_rank,.52*risk_rank)
    np.testing.assert_array_equal(expected_core,original['core']);np.testing.assert_array_equal(expected_core,tr['core'])
    state=state_oracle(frame,meta,sessions)
    compare(r.json(p/'STATE_AUDIT.json'),state['audit'],'STATE_AUDIT')
    require(a['directional_q'].shape==(122,) and a['directional_q'].dtype.kind=='f','State array schema')
    np.testing.assert_allclose(a['directional_q'],state['q'],rtol=0,atol=STATE_ATOL,equal_nan=True)
    require(np.array_equal(np.isnan(a['directional_q']),np.isnan(state['q'])),'Availability mask')
    scores,choices,deltas=score_oracle(a,meta,raw_rank,risk_rank,state['q'])
    saved_scores=r.npz(p/'SCORES.npz');saved_orders=r.json(p/'PRETRAIN_ORDERS.json');require(set(saved_scores)==set(saved_orders)==set(NAMES),'Three output arms')
    wanted={}
    for name in NAMES:
        wanted[name]=[dict(date=day,signal_date=day,end=meta['ends'][i],symbols=[meta['symbols'][i][j] for j in choice],
            scores=[float(scores[name][i,j]) for j in choice]) for i,(day,choice) in enumerate(zip(meta['dates'],choices[name]))]
        compare(saved_orders[name],wanted[name],'orders/'+name)
        np.testing.assert_allclose(saved_scores[name],scores[name],rtol=0,atol=SCORE_ATOL)
    h.assert_exact_tree(saved_orders[NAMES[0]],r.json(inputs['original_orders']))
    require(saved_scores[NAMES[0]].tobytes()==scores[NAMES[0]].tobytes(),'Baseline score bytes')
    for n in (0,1,36,53,60,76,88,122):
        cut=frame.loc[:meta['dates'][n-1]] if n else frame.iloc[:0]
        subset=dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n])
        prefix=state_oracle(cut,subset,sessions)
        np.testing.assert_array_equal(prefix['q'],state['q'][:n])
        compare(prefix['audit']['months'],state['audit']['months'][:n],'State prefix')
        ss,cc,_=score_oracle(a,meta,raw_rank,risk_rank,prefix['q'],n)
        for name in NAMES:require(cc[name]==choices[name][:n] and ss[name].tobytes()==scores[name][:n].tobytes(),'Score/choices prefix')
    missing=np.full(122,np.nan);ss,cc,dd=score_oracle(a,meta,raw_rank,risk_rank,missing)
    for name in NAMES[1:]:require(ss[name].tobytes()==scores[NAMES[0]].tobytes() and cc[name]==choices[NAMES[0]],'Missing exact fallback')
    expected_feature=dict(original_arrays_unchanged=True,transition_cache_lineage_verified=True,original_months=122,original_width=49,
        new_model_fits=state['audit']['new_model_fits'],returns_or_nav_computed=False,original_auxiliary_states_and_core_coefficients_preserved=True)
    compare(r.json(p/'FEATURE_AUDIT.json'),expected_feature,'FEATURE_AUDIT')
    required={str(p/x):r.sha(p/x) for x in (*CACHE_FILES,'PLAN.json','CACHE.json')};allowed=dict(pins,**required)
    for key,value in list(r.seen.items()):require(r.sha(key)==value and allowed.get(key)==value,'Unbound consumed artifact '+key)
    return dict(schema_version=1,passed=True,stage='prepare',plan_sha256=plan_sha,cache_sha256=r.sha(p/'CACHE.json'),
        orders_sha256=r.sha(p/'PRETRAIN_ORDERS.json'),**{k:True for k in CHECKS},source_hashes=sources,artifact_hashes=dict(r.seen),
        producer_modules_imported=False,returns_or_nav_computed=False,monthly_label_models_fit=0,
        independent_daily_model_fits=0,new_model_fits=0,orders_checked=366,
        state_months_checked=122,stock_windows_checked=122*32,
        available_state_months=int(np.isfinite(state['q']).sum()),
        max_state_error=float(np.nanmax(np.abs(a['directional_q']-state['q']))) if np.isfinite(state['q']).any() else 0.,
        max_score_error=max(float(np.max(np.abs(saved_scores[k]-scores[k]))) for k in NAMES),raw_prices_independently_verified=True,
        calendar_overlay_independently_verified=True,price_parity=parity,all_signal_prefixes_verified=True,
        zero_state_is_valid_not_missing=True,zero_dispersion_unavailable=True,missing_state_exact_baseline=True,
        limitations=['Decimal state reconstruction and original-core accounting only; no model fit or strategy return is computed.',
            'Original proxy universe, current-vintage adjusted prices and reused historical sample limitations remain.'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--plan',type=Path,default=ROOT/STUDY/'PLAN.json');parser.add_argument('--prepare-complete',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/STUDY/'INDEPENDENT_PREPARE.json');args=parser.parse_args()
    require(args.prepare_complete,'Run only after registered preparation')
    result=audit(ROOT,args.plan,args.plan_sha)
    with args.output.open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,output=str(args.output),sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__=='__main__':main()
