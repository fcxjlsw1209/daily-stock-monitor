"""Independent round138 conditional sector-moment and preparation review.

Only explicitly invoked audit() loads real caches. Conditional moments use
Decimal arithmetic; no round138 producer module is imported by this checker.
"""
from pathlib import Path
from decimal import Decimal, localcontext
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
STUDY='outputs/monthly_sector_asymmetry138'
LOCAL=old.LOCAL
PROPOSAL_SHA='70d04635a591b4bf44de7d9c8574232d22d195edb905d8dff080a367fe8d30e7'
CALENDAR_SHA='4b8f32ad8a555d6537d7cf3bd0f831e64fbb2de26d033c8daaf68590fcd4a6ea'
MAPPING='outputs/monthly_strict_relative/strict_study.py'
MAPPING_SHA='ddb11c0149e920de2952daba515ce28625a56df1f5c1039d4b9920c6d58d02c6'
NAMES=('incumbent_local67','sector_beta_asymmetry','sector_correlation_asymmetry')
INPUTS=('proposal','meta','policy','original_orders','prices','sessions','protected_manifest')
CHECKS=('original27_arrays_byte_identical','all_features_independently_reconstructed',
        'all_three_score_matrices_and_ordered_choices_independently_verified',
        'one_episode_and_prefix_isolation_verified','zero_and_missing_exact_fallback')
EXTRA=('raw_beta_gap','raw_correlation_gap','rank_beta_gap','rank_correlation_gap','common_available')
CACHE_FILES=('DATA.npz','SCORES.npz','META.json','FEATURE_AUDIT.json','PRETRAIN_ORDERS.json','PRICE_PARITY.json')
ATOL=1e-12
require=old.require
SECTOR={'Basic Materials':'XLB','Communication Services':'XLC','Energy':'XLE','Financial Services':'XLF',
        'Industrials':'XLI','Technology':'XLK','Consumer Defensive':'XLP','Real Estate':'XLRE',
        'Utilities':'XLU','Healthcare':'XLV','Consumer Cyclical':'XLY'}
FOREIGN={'ASML':'XLK','ARM':'XLK','TSM':'XLK','BABA':'XLY','NVO':'XLV','SAP':'XLK','TM':'XLY',
         'SHEL':'XLE','SKHY':'XLK','BHP':'XLB','HSBC':'XLF','SMFG':'XLF','NVS':'XLV','SNDK':'XLK'}


def source_files(root=None):
    root=ROOT if root is None else Path(root).resolve()
    return sorted(set([root/'work/monthly_sector_asymmetry138_prepare_review.py',
                       root/'work/test_monthly_sector_asymmetry138_prepare_review.py',root/MAPPING,root/'outputs/monthly_path70_research/incumbent_adapter.py']+old.source_files(root)))


def verify_mapping_source(reader):
    require(reader.sha(MAPPING)==MAPPING_SHA,'Original mapping code')
    tree=ast.parse(reader.path(MAPPING).read_text());found={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name)and target.id in ('SECTOR','FOREIGN'):found[target.id]=ast.literal_eval(node.value)
    require(found==dict(SECTOR=SECTOR,FOREIGN=FOREIGN),'Original mapping constants')


def benchmark_map(date,names,sectors,etfs):
    require(isinstance(sectors,pd.DataFrame)and sectors.index.is_unique and sectors.columns.is_unique,'Sector source axes')
    out={};day=pd.Timestamp(date)
    for s in names:
        if s in etfs:out[s]=None
        elif s in FOREIGN:out[s]=FOREIGN[s]
        elif s in sectors and day in sectors.index:out[s]=SECTOR.get(sectors.at[day,s])
        else:out[s]=None
    return out


def finite_vector(values,n):
    a=np.asarray(values)
    require(a.shape==(n,),'Vector length')
    require(all(isinstance(x,(float,int,np.floating,np.integer))and not isinstance(x,(bool,np.bool_))for x in a),'Numeric nonboolean vector')
    out=[float(v)for v in a];require(all(math.isfinite(v)for v in out),'Finite vector');return out


def group_moments(stock,sector):
    y=finite_vector(stock,len(stock));x=finite_vector(sector,len(stock));require(len(x)>=2,'Group size')
    with localcontext() as c:
        c.prec=180
        dx=[Decimal.from_float(v)for v in x];dy=[Decimal.from_float(v)for v in y]
        mx=sum(dx)/len(dx);my=sum(dy)/len(dy);dx=[v-mx for v in dx];dy=[v-my for v in dy]
        xx=sum(v*v for v in dx);yy=sum(v*v for v in dy);xy=sum(a*b for a,b in zip(dx,dy))
        if xx==0 or yy==0:return None
        beta=float(xy/xx);rho=float(xy/(xx*yy).sqrt())
    require(math.isfinite(beta)and math.isfinite(rho)and -1<=rho<=1,'Representable moments')
    return dict(beta=beta,rho=rho)


def moments_oracle(stock,sector):
    y=finite_vector(stock,252);x=finite_vector(sector,252)
    positive=[i for i,v in enumerate(x)if v>0];negative=[i for i,v in enumerate(x)if v<0]
    counts=dict(positive_count=len(positive),negative_count=len(negative),zero_count=252-len(positive)-len(negative))
    if min(len(positive),len(negative))<40:return dict(available=False,reason='insufficient_group',**counts)
    up=group_moments([y[i]for i in positive],[x[i]for i in positive]);down=group_moments([y[i]for i in negative],[x[i]for i in negative])
    if up is None or down is None:return dict(available=False,reason='constant_group',**counts)
    beta=up['beta']-down['beta'];rho=up['rho']-down['rho']
    require(math.isfinite(beta)and math.isfinite(rho),'Finite group differences')
    return dict(available=True,reason='available',beta_up=up['beta'],beta_down=down['beta'],rho_up=up['rho'],rho_down=down['rho'],raw_beta_gap=beta,raw_rho_gap=rho,**counts)


def rank_oracle(values):
    out=np.full(len(values),np.nan);ids=[i for i,v in enumerate(values)if math.isfinite(v)]
    for i in ids:
        below=sum(values[j]<values[i]for j in ids);equal=sum(values[j]==values[i]for j in ids)
        out[i]=(below+(equal+1)/2)/len(ids)-.5
    return out


def score_oracle(a,meta,beta_hand,rho_hand,n=None):
    h=old.original_helper();x=h.original_prefix(a,meta,n);n=x['n'];shape=x['core'].shape
    hands=[np.asarray(hand)[:n]for hand in (beta_hand,rho_hand)]
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
        require(type(actual)in (float,int)and math.isfinite(actual)and math.isclose(actual,expected,abs_tol=ATOL,rel_tol=0),'Numeric '+path)
    else:require(type(actual)is type(expected)and actual==expected,'Discrete '+path)


def log_returns(prices):
    p=finite_vector(prices,253);require(all(v>0 for v in p),'Positive prices')
    # The proposal fixes this binary64 log vector before independent moments.
    m,e=np.frexp(np.asarray(p,dtype=float))
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        values=np.log(m[1:]/m[:-1])+np.diff(e)*np.log(2.)
    require(np.isfinite(values).all(),'Finite log returns');return values


def detailed_group(stock,sector):
    """Decimal centered sufficient statistics, independent of scaled float fitting."""
    y=finite_vector(stock,len(stock));x=finite_vector(sector,len(stock))
    with localcontext() as context:
        context.prec=180
        yy=[Decimal.from_float(v) for v in y];xx=[Decimal.from_float(v) for v in x]
        my=sum(yy)/len(yy);mx=sum(xx)/len(xx)
        cy=[v-my for v in yy];cx=[v-mx for v in xx]
        sy=max(map(abs,cy));sx=max(map(abs,cx))
        zy=[v/sy for v in cy] if sy else [Decimal(0)]*len(y)
        zx=[v/sx for v in cx] if sx else [Decimal(0)]*len(x)
        yss=sum(v*v for v in zy);xss=sum(v*v for v in zx);cross=sum(a*b for a,b in zip(zx,zy))
        beta=float(cross/xss*sy/sx) if sx and sy else None
        rho=float(cross/(xss*yss).sqrt()) if sx and sy else None
        result=dict(count=len(x),stock_mean=float(my),sector_mean=float(mx),stock_scale=float(sy),sector_scale=float(sx),
            stock_ss_scaled=float(yss),sector_ss_scaled=float(xss),cross_ss_scaled=float(cross),beta=beta,rho=rho)
    require(all(v is None or math.isfinite(v) for v in result.values()),'Finite moments')
    require((not sy or result['stock_scale']>0) and (not sx or result['sector_scale']>0),'Representable nonzero group scale')
    return result


def empty_detail(reason):
    return dict(available=False,reason=reason,up=None,down=None,zero_count=None,raw_beta_gap=None,raw_correlation_gap=None)


def detailed_moments(stock,sector):
    y=finite_vector(stock,252);x=finite_vector(sector,252)
    groups=([i for i,v in enumerate(x) if v>0],[i for i,v in enumerate(x) if v<0])
    result=empty_detail('insufficient_group');result['zero_count']=sum(v==0 for v in x)
    if min(map(len,groups))<40:return result
    for name,indices in zip(('up','down'),groups):
        result[name]=detailed_group([y[i] for i in indices],[x[i] for i in indices])
    for name in ('up','down'):
        if result[name]['sector_scale']==0:result['reason']='zero_sector_variance';return result
        if result[name]['stock_scale']==0:result['reason']='zero_stock_variance';return result
    result.update(available=True,reason='available',raw_beta_gap=result['up']['beta']-result['down']['beta'],
        raw_correlation_gap=result['up']['rho']-result['down']['rho'])
    require(math.isfinite(result['raw_beta_gap']) and math.isfinite(result['raw_correlation_gap']),'Finite gap')
    return result


def price_vector(values):
    a=np.asarray(values)
    require(a.shape==(253,) and all(isinstance(v,(float,int,np.floating,np.integer)) and not isinstance(v,(bool,np.bool_)) for v in a),'Numeric price path')
    return np.asarray(a,dtype=float)


def conditional_cells(panel,meta,sessions):
    prices=panel['adj'];sectors=panel['sectors'];calendar=pd.DatetimeIndex(sessions)
    require(isinstance(prices,pd.DataFrame) and prices.columns.is_unique,'Unique price columns')
    for axis in (pd.DatetimeIndex(prices.index),calendar,pd.DatetimeIndex(meta['dates'])):
        require(not axis.hasnans and axis.tz is None and axis.is_unique and axis.is_monotonic_increasing
                and axis.equals(axis.normalize()),'Ordered canonical daily axes')
    dates=meta['dates'];require(len(dates)==len(meta['symbols']) and len(set(meta['etfs']))==len(meta['etfs'])==17,'Original metadata')
    require(len(calendar)>0,'Nonempty calendar')
    if len(dates):
        axis=prices.index[(prices.index>=calendar[0]) & (prices.index<=pd.Timestamp(dates[-1]))]
        require(axis.isin(calendar).all(),'Official source price dates')
    out=[]
    for i,(day,names) in enumerate(zip(dates,meta['symbols'])):
        require(len(names)==len(set(names))==49 and set(meta['etfs'])<=set(names),'Original49 pool')
        day=pd.Timestamp(day);require(day in calendar,'Official signal date');pos=calendar.get_loc(day)
        window=calendar[pos-294:pos-41] if pos>=294 else calendar[:0]
        mapping=benchmark_map(day,names,sectors,meta['etfs'])
        for j,symbol in enumerate(names):
            bench=mapping[symbol];detail=empty_detail('etf_not_applicable')
            if symbol not in meta['etfs']:
                if bench is None:detail=empty_detail('missing_sector_mapping')
                elif len(window)!=253:detail=empty_detail('insufficient_history')
                else:
                    # Check each physical path without compressing absent session rows.
                    x=price_vector(prices.reindex(index=window,columns=[bench]).to_numpy()[:,0])
                    y=price_vector(prices.reindex(index=window,columns=[symbol]).to_numpy()[:,0])
                    if not (np.isfinite(x) & (x>0)).all():detail=empty_detail('invalid_sector_path')
                    elif not (np.isfinite(y) & (y>0)).all():detail=empty_detail('invalid_stock_path')
                    else:detail=detailed_moments(log_returns(y),log_returns(x))
            out.append(dict(signal_index=i,pool_index=j,signal_date=str(day.date()),symbol=symbol,benchmark=bench,
                window_start=str(window[0].date()) if len(window) else None,
                window_end=str(window[-1].date()) if len(window) else None,**detail,
                rank_beta_gap=None,rank_correlation_gap=None))
    return out


def features_oracle(panel,meta,sessions,validated_raw=None):
    cells=conditional_cells(panel,meta,sessions);shape=(len(meta['dates']),49)
    arrays={key:np.full(shape,np.nan) for key in EXTRA[:-1]};arrays[EXTRA[-1]]=np.zeros(shape,bool)
    for cell in cells:
        i,j=cell['signal_index'],cell['pool_index']
        arrays['common_available'][i,j]=cell['available']
        if cell['available']:
            for key in EXTRA[:2]:arrays[key][i,j]=cell[key]
    if validated_raw is not None:
        for key in EXTRA[:2]:
            ref=validated_raw[key]
            require(ref.shape==shape and ref.dtype==np.dtype(float),'Raw feature schema')
            require(np.array_equal(np.isfinite(ref),arrays['common_available']) and not np.isinf(ref).any(),'Exact raw availability')
            np.testing.assert_allclose(ref,arrays[key],atol=ATOL,rtol=0,equal_nan=True)
    for i in range(shape[0]):
        for raw,rank in zip(EXTRA[:2],EXTRA[2:4]):
            # Ranking uses exact verified binary64 raw values, as preregistered.
            values=arrays[raw][i] if validated_raw is None else validated_raw[raw][i]
            arrays[rank][i]=rank_oracle(values)
            for j,value in enumerate(arrays[rank][i]):
                if math.isfinite(value):cells[i*49+j][rank]=float(value)
    return dict(arrays=arrays,audit=dict(original_arrays_unchanged=True,original_months=shape[0],original_width=49,
        new_model_fits=0,returns_or_nav_computed=False,original_auxiliary_states_and_core_coefficients_preserved=True,
        mapping_source_sha256=MAPPING_SHA,cells=cells))


def audit(root,plan_path,plan_sha):
    require(re.fullmatch('[0-9a-f]{64}',plan_sha) is not None,'Explicit PLAN SHA')
    r=old.Reader(root);p=r.path(STUDY);require(r.path(plan_path)==p/'PLAN.json','Dedicated PLAN')
    require(r.sha(plan_path)==plan_sha,'PLAN SHA');plan=r.json(plan_path)
    require(plan.get('round')==138 and plan.get('status')=='REGISTERED' and plan.get('arms')==list(NAMES)
            and plan.get('primary_challenger')==NAMES[2],'Round138 policy')
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
    spec=importlib.util.spec_from_file_location('sector138_raw_loader',r.path(old.LOADER))
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    panel,info=loader.load_panel(meta['etfs']);used=dict(info['source_hashes']);used[str(r.path(old.LOADER))]=old.LOADER_SHA
    require({str(r.path(k)):v for k,v in used.items()}==raw,'Actually consumed raw sources');r.verify(used)
    verify_mapping_source(r)
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
    beta=expected['arrays']['rank_beta_gap'];rho=expected['arrays']['rank_correlation_gap']
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
        ss,cc,_=score_oracle(a,meta,prefix['arrays']['rank_beta_gap'],prefix['arrays']['rank_correlation_gap'],n)
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
        original_sector_map_independently_reproduced=True,all_signal_prefixes_verified=True,
        raw_zero_uses_ordinary_average_rank=True,ranked_zero_and_missing_exact_baseline=True,
        limitations=['Decimal conditional moments verify the declared float64 log-return vectors; exact ranks use validated producer binary64 raw values.',
            'No monthly models or strategy returns computed; inherited proxy-universe, sector-classification and price-vintage limitations remain.'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--plan',type=Path,default=ROOT/STUDY/'PLAN.json');parser.add_argument('--prepare-complete',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/STUDY/'INDEPENDENT_PREPARE.json');args=parser.parse_args()
    require(args.prepare_complete,'Run only after registered preparation')
    result=audit(ROOT,args.plan,args.plan_sha)
    with args.output.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(passed=True,output=str(args.output),sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__=='__main__':main()
