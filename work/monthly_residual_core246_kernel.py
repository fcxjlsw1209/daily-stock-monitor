"""Pure lagged market/joint descriptive OLS and matched half-core integration.

No loaders, actual-data admission, predictive training or portfolio returns.
The caller authenticates sources and enforces visible security listing bounds.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import math
from numbers import Real
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
AXES_PATH=ROOT/'work/monthly_risk_consensus219_kernel.py'
AXES_SHA='7316b7cc081e7f8c25fc8cbeb51f43642223367609be1f77e4beb81929c1be77'
ADAPTER_PATH=ROOT/'outputs/monthly_core_timing70_research/core_adapter.py'
ADAPTER_SHA='b1c45a7e9fc671675b461e6f016eda4e17a7885c1ff74630cadddf9ce8d1544a'

def _module(path,digest,name):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Changed frozen dependency: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

axes=_module(AXES_PATH,AXES_SHA,'residual246_axes219')
core_adapter=_module(ADAPTER_PATH,ADAPTER_SHA,'residual246_core_adapter')
ARMS=('incumbent_local67','joint_residual_halfcore','market_residual_halfcore_control')
VALUE_KEYS=('available','candidate','control')
FEATURE_KEYS=VALUE_KEYS
COUNT_FIELDS=('events','symbols_per_event','checked_cells','stock_cells','etf_cells','stock_source_available_cells',
    'market_source_available_cells','sector_source_available_cells','source_available_cells','available_cells','unavailable_cells',
    'stock_price_observations','market_price_observations','sector_price_observations','rolling_windows_checked',
    'descriptive_market_estimates','descriptive_joint_estimates','ranked_cells','new_model_fits','predictive_model_fits')
MAPPING_FIELDS=('query_index','column_index','date','symbol','issuer_key','original_sector','sector','mapping_origin','mapping_repair','mapping_evidence')
FIT_KEYS=('market_ss','sector_ss','rho','conditioning','fit_available','beta_market_only','beta_market_joint','beta_sector_joint',
    'adjusted_market','adjusted_joint','response_ss','fit_reason')
MEASURE_KEYS=('stock_source_available','market_source_available','sector_source_available','stock_source_reason','market_source_reason',
    'sector_source_reason','source_available','available','reason','stock_start_price','stock_end_price','market_start_price','market_end_price',
    'sector_start_price','sector_end_price','stock_mean','stock_square_sum','stock_sample_vol','market_adjusted_sum','joint_adjusted_sum',
    'market_adjusted_mean','joint_adjusted_mean','raw_candidate','raw_control','fit_arrays')
CELL_KEYS=('column_index','symbol','is_etf','mapping')+MEASURE_KEYS+('candidate','control')
EVENT_KEYS=('query_index','date','symbols','session_position','price_start','price_end','source_dates','fit_dates',
    'missing_official_sessions','missing_vox_sessions','stock_source_available_indices','available_indices','cells')
DECISION_COUNT_FIELDS=('events','symbols_per_event','active_events','available_active_cells','candidate_nonzero_delta_cells',
    'control_nonzero_delta_cells','original_episode_calls','new_model_fits','predictive_model_fits')
DECISION_EVENT_KEYS=('query_index','date','initial_partial','calm','weak','active','available_indices','extension_active_indices',
    'episode_budget','candidate_delta','control_delta','original_remaining_support','candidate_remaining_support','control_remaining_support')
PRICE_COUNT=505
WINDOW=252
CONDITION_MIN=1e-8
ATOL=2e-10
RTOL=2e-10
INITIAL_PARTIAL='2016-08-17'
POLICY=dict(budget=.05,ceiling=5,gap=.02)
BLEND=.5

def require(ok,message):
    if not ok:raise ValueError(message)

def dependency_files():
    return (AXES_PATH,ADAPTER_PATH,core_adapter.PATH_ADAPTER_PATH,ROOT/'outputs/monthly_local_linear70/study.py',
        ROOT/'outputs/monthly_daily_episodes/episodes.py',ROOT/'outputs/monthly_return_priority/return_priority.py',
        ROOT/'outputs/monthly_local_linear70/INCUMBENT_POLICY.json')

def _finite(value):
    value=float(value)
    if not math.isfinite(value):raise FloatingPointError('Nonfinite derived arithmetic')
    return value

def _sum(values):return _finite(math.fsum(values))

def _divide(a,b):
    v=_finite(a/b)
    if a!=0. and v==0.:raise FloatingPointError('Division underflow')
    return v

def _squares(values):
    squares=[_finite(v*v) for v in values]
    if any(v!=0. and q==0. for v,q in zip(values,squares)):raise FloatingPointError('Centered square underflow')
    return _sum(squares)

def _center(values):
    mean=_divide(_sum(values),float(len(values)))
    centered=[_finite(x-mean) for x in values]
    return mean,centered,_squares(centered)

def _dot(a,b):return _sum(_finite(x*y) for x,y in zip(a,b))

def _empty_source(reason):return dict(available=False,reason=reason,prices=None)

def _source(prices):
    require(np.shape(prices)==(PRICE_COUNT,),'Exactly505 adjusted prices required')
    raw=np.asarray(prices,dtype=object)
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Real) for v in raw):return _empty_source('nonnumeric_adjusted_price')
    try:out=[float(v) for v in raw]
    except (TypeError,ValueError,OverflowError):return _empty_source('nonnumeric_adjusted_price')
    if not all(math.isfinite(v) and v>0. for v in out):return _empty_source('invalid_adjusted_price')
    return dict(available=True,reason='available',prices=out)

def _returns(prices):return [_finite(_divide(prices[k],prices[k-1])-1.) for k in range(1,PRICE_COUNT)]

def _fit(y,m,s):
    """Centered, norm-scaled reduced QR; no inverse or raw rolling moments."""
    _,yc,yss=_center(y);_,mc,mss=_center(m);_,sc,sss=_center(s)
    rho=condition=None
    if mss==0.:reason='zero_market_variance'
    elif sss==0.:reason='zero_sector_variance'
    else:
        mn=_finite(math.sqrt(mss));sn=_finite(math.sqrt(sss))
        rho=_divide(_dot(mc,sc),_finite(mn*sn));condition=_finite(1.-_finite(rho*rho))
        reason='available' if condition>CONDITION_MIN else 'collinear_factors'
    result=dict(market_ss=mss,sector_ss=sss,rho=rho,conditioning=condition,fit_available=reason=='available',
        beta_market_only=None,beta_market_joint=None,beta_sector_joint=None,adjusted_market=None,adjusted_joint=None,
        response_ss=yss,fit_reason=reason)
    if reason!='available':return result
    u=[_divide(x,mn) for x in mc];v=[_divide(x,sn) for x in sc]
    design=np.asarray([u,v],float).T
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        q,r=np.linalg.qr(design,mode='reduced')
    require(q.shape==(WINDOW,2) and r.shape==(2,2),'Reduced QR axes')
    if not np.isfinite(q).all() or not np.isfinite(r).all():raise FloatingPointError('Nonfinite QR')
    rhs=np.array([_dot(q[:,j],yc) for j in range(2)])
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):scaled=np.linalg.solve(r,rhs)
    except np.linalg.LinAlgError as exc:raise FloatingPointError('Admitted QR singular') from exc
    bm=_divide(_dot(u,yc),mn);bjm=_divide(_finite(scaled[0]),mn);bjs=_divide(_finite(scaled[1]),sn)
    market_error=[_finite(y-bm*x) for y,x in zip(yc,mc)]
    joint_error=[_finite(_finite(y-bjm*x)-bjs*z) for y,x,z in zip(yc,mc,sc)]
    tol=ATOL+RTOL*_finite(math.sqrt(yss))
    if any(abs(x)>tol for x in (_dot(u,market_error),_dot(u,joint_error),_dot(v,joint_error))):
        raise FloatingPointError('Scaled normal-equation residual exceeds frozen tolerance')
    result.update(beta_market_only=bm,beta_market_joint=bjm,beta_sector_joint=bjs)
    return result

def _measure(stock,market,sector,mapped=True,etf=False):
    sources=(stock,market,sector)
    result=dict.fromkeys(MEASURE_KEYS,None)
    for name,src in zip(('stock','market','sector'),sources):
        result[name+'_source_available']=src['available'];result[name+'_source_reason']=src['reason']
        if src['available']:result[name+'_start_price']=src['prices'][0];result[name+'_end_price']=src['prices'][-1]
    sourcegood=all(src['available'] for src in sources) and mapped and not etf
    result.update(source_available=sourcegood,available=False)
    if etf:result['reason']='etf_unchanged'
    elif not stock['available']:result['reason']='stock_'+stock['reason']
    elif not mapped:result['reason']='unresolved_mapping'
    elif not market['available']:result['reason']='market_'+market['reason']
    elif not sector['available']:result['reason']='sector_'+sector['reason']
    if not sourcegood:return result
    try:
        y,m,s=(_returns(src['prices']) for src in sources)
        mean,_,ss=_center(y[WINDOW:]);sd=_finite(math.sqrt(_divide(ss,float(WINDOW-1))))
        result.update(stock_mean=mean,stock_square_sum=ss,stock_sample_vol=sd)
        arrays={key:[] for key in FIT_KEYS}
        for k in range(WINDOW,2*WINDOW):
            fit=_fit(y[k-WINDOW:k],m[k-WINDOW:k],s[k-WINDOW:k])
            if fit['fit_available']:
                fit['adjusted_market']=_finite(y[k]-_finite(fit['beta_market_only']*m[k]))
                fit['adjusted_joint']=_finite(_finite(y[k]-_finite(fit['beta_market_joint']*m[k]))-_finite(fit['beta_sector_joint']*s[k]))
            for key in FIT_KEYS:arrays[key].append(fit[key])
        result['fit_arrays']=arrays
        allfits=all(arrays['fit_available'])
        if allfits:
            ms=_sum(arrays['adjusted_market']);js=_sum(arrays['adjusted_joint'])
            mm=_divide(ms,float(WINDOW));jm=_divide(js,float(WINDOW))
            result.update(market_adjusted_sum=ms,joint_adjusted_sum=js,market_adjusted_mean=mm,joint_adjusted_mean=jm)
        if sd==0.:result['reason']='zero_stock_sample_volatility'
        elif not allfits:result['reason']=next(r for r in arrays['fit_reason'] if r!='available')
        else:result.update(available=True,reason='available',raw_candidate=_divide(jm,sd),raw_control=_divide(mm,sd))
    except (OverflowError,ZeroDivisionError) as exc:raise FloatingPointError('Derived arithmetic failure') from exc
    return result

def measure(stock,market,sector):
    """Three505-price windows, with complete252 lagged descriptive-fit audit."""
    return _measure(_source(stock),_source(market),_source(sector))

def _rank(raw,available):
    indices=np.flatnonzero(available).tolist();out=np.full(len(raw),np.nan);n=len(indices)
    for j in indices:
        require(math.isfinite(raw[j]),'Finite available raw rank')
        less=sum(raw[k]<raw[j] for k in indices);equal=sum(raw[k]==raw[j] for k in indices)
        out[j]=(less+(equal+1)/2.)/n-.5
    return out

def _inputs(panel,meta,calendar,source_review,limit):
    require(isinstance(panel,dict) and set(panel)=={'adj','vox'},'Panel requires exactly adj and vox')
    for frame in panel.values():axes._frame(frame)
    require(isinstance(meta,dict) and all(k in meta for k in ('dates','symbols','etfs')),'Feature metadata')
    require(not isinstance(meta['dates'],(str,bytes)) and not isinstance(meta['symbols'],(str,bytes)),'Metadata sequences')
    total=len(meta['dates']);require(len(meta['symbols'])==total,'Metadata date/symbol axes')
    n=total if limit is None else limit;require(type(n) is int and 0<=n<=total,'Valid prefix')
    require(isinstance(meta['etfs'],(list,tuple)),'Explicit ETF list')
    require(not isinstance(calendar,(str,bytes)),'Explicit calendar')
    sessions=axes._calendar(calendar) if len(calendar) else pd.DatetimeIndex([])
    consumed=dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n]);queries,width=axes._queries(consumed,sessions)
    if n==0 and total:
        require(not isinstance(meta['symbols'][0],(str,bytes)),'Width sequence');width=len(meta['symbols'][0])
    require(isinstance(source_review,dict) and isinstance(source_review.get('mapping_records'),list),'Mapping records required')
    etfs=set(meta['etfs']);records={};issuers={i:set() for i in range(n)}
    for record in source_review['mapping_records']:
        require(isinstance(record,dict) and type(record.get('query_index')) is int and record['query_index']>=0,'Mapping query index')
        i=record['query_index']
        if i>=n:continue
        require(set(record)==set(MAPPING_FIELDS),'Exact ten mapping fields')
        day,names,_=queries[i];j=record['column_index']
        require(type(j) is int and 0<=j<len(names) and (i,j) not in records,'Unique in-domain mapping cell')
        require(record['date']==str(day.date()) and record['symbol']==names[j] and names[j] not in etfs,'Exact direct-stock mapping identity')
        issuer=record['issuer_key'];require(type(issuer) is str and bool(issuer) and issuer.strip()==issuer and issuer not in issuers[i],'Unique nonempty issuer')
        for field in ('original_sector','sector'):
            value=record[field];require(value is None or (type(value) is str and bool(value) and value.strip()==value),'Sector identity')
        require(all(type(record[k]) is str and bool(record[k]) for k in ('mapping_origin','mapping_repair')),'Mapping provenance')
        evidence=record['mapping_evidence'];require(evidence is None or isinstance(evidence,(list,str,dict)),'Mapping evidence metadata')
        records[i,j]={k:deepcopy(record[k]) for k in MAPPING_FIELDS};issuers[i].add(issuer)
    expected={(i,j) for i,(_,names,_) in enumerate(queries) for j,name in enumerate(names) if name not in etfs}
    require(set(records)==expected,'Exact complete direct-stock mapping inventory')
    return sessions,queries,width,records,etfs

def _extract(frame,dates,missing,symbol):
    if not len(dates):return _empty_source('insufficient_history')
    if symbol not in frame.columns:return _empty_source('missing_symbol')
    if missing:return _empty_source('missing_official_session')
    return _source(frame.loc[dates,symbol].to_numpy(copy=True))

def build_features(panel,meta,calendar,source_review,limit=None):
    sessions,queries,width,records,etfs=_inputs(panel,meta,calendar,source_review,limit);shape=(len(queries),width)
    values={key:np.zeros(shape,bool) if key=='available' else np.full(shape,np.nan) for key in VALUE_KEYS}
    counts=dict.fromkeys(COUNT_FIELDS,0);counts.update(events=len(queries),symbols_per_event=width,checked_cells=len(queries)*width)
    history=[];reasons=Counter();adj=panel['adj'];vox=panel['vox']
    for i,(day,names,t) in enumerate(queries):
        dates=sessions[t-546:t-41] if t>=546 else pd.DatetimeIndex([])
        isos=dates.strftime('%Y-%m-%d').tolist();missing=dates.difference(adj.index).strftime('%Y-%m-%d').tolist()
        vm=dates.difference(vox.index).strftime('%Y-%m-%d').tolist();cells=[];source_indices=[]
        for j,symbol in enumerate(names):
            etf=symbol in etfs;mapping=records.get((i,j))
            if etf:
                empty=_empty_source('etf_unchanged');result=_measure(empty,empty,empty,False,True)
            else:
                stock=_extract(adj,dates,missing,symbol);market=_extract(adj,dates,missing,'SPY');benchmark=mapping['sector']
                sector=(_extract(vox,dates,vm,'VOX') if benchmark=='VOX' else _extract(adj,dates,missing,benchmark)) if benchmark else _empty_source('unresolved_mapping')
                result=_measure(stock,market,sector,benchmark is not None)
            values['available'][i,j]=result['available'];reasons[result['reason']]+=1
            counts['etf_cells' if etf else 'stock_cells']+=1
            for name in ('stock','market','sector'):
                counts[name+'_source_available_cells']+=int(result[name+'_source_available'])
            counts['source_available_cells']+=int(result['source_available']);counts['available_cells']+=int(result['available'])
            if result['stock_source_available']:source_indices.append(j)
            if result['fit_arrays'] is not None:
                number=sum(result['fit_arrays']['fit_available']);counts['descriptive_market_estimates']+=number;counts['descriptive_joint_estimates']+=number
            cell=dict(column_index=j,symbol=symbol,is_etf=etf,mapping=mapping,**result,candidate=None,control=None)
            require(tuple(cell)==CELL_KEYS,'Cell schema');cells.append(cell)
        for key,rawkey in (('candidate','raw_candidate'),('control','raw_control')):
            values[key][i]=_rank([c[rawkey] for c in cells],values['available'][i])
        available_indices=np.flatnonzero(values['available'][i]).tolist()
        for j in available_indices:
            cells[j]['candidate']=float(values['candidate'][i,j]);cells[j]['control']=float(values['control'][i,j])
        fit_dates=[dict(training_price_start=isos[d-253],training_return_start=isos[d-252],training_return_end=isos[d-1],application_date=isos[d]) for d in range(253,505)] if isos else []
        event=dict(query_index=i,date=str(day.date()),symbols=names,session_position=t,price_start=isos[0] if isos else None,
            price_end=isos[-1] if isos else None,source_dates=isos,fit_dates=fit_dates,missing_official_sessions=missing,
            missing_vox_sessions=vm,stock_source_available_indices=source_indices,available_indices=available_indices,cells=cells)
        require(tuple(event)==EVENT_KEYS,'Event schema');history.append(event)
    for name in ('stock','market','sector'):counts[name+'_price_observations']=PRICE_COUNT*counts[name+'_source_available_cells']
    counts.update(unavailable_cells=counts['checked_cells']-counts['available_cells'],rolling_windows_checked=WINDOW*counts['source_available_cells'],ranked_cells=2*counts['available_cells'])
    return values,dict(schema='monthly_residual_core246_features_v1',events=len(queries),counts=counts,reason_counts=dict(reasons),event_history=history)


def _snapshot(arrays):
    return {k:(np.asarray(v).shape,np.asarray(v).dtype.str,np.asarray(v).tobytes()) for k,v in arrays.items()}


def _numeric(value,shape,n,label,finite=True):
    raw=np.asarray(value);require(raw.shape==shape,'Exact shape: '+label)
    require(all(isinstance(v,Real) and not isinstance(v,(bool,np.bool_)) for v in raw[:n].flat),'Real nonboolean: '+label)
    try:out=raw[:n].astype(float,copy=True)
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError('Unrepresentable '+label) from exc
    if finite:require(np.isfinite(out).all(),'Finite: '+label)
    return out


def _bool(value,shape,n,label):
    a=np.asarray(value);require(a.shape==shape and a.dtype.kind=='b','Boolean shape: '+label);return a[:n].copy()


def _features(values,shape,n):
    require(isinstance(values,dict) and set(values)==set(VALUE_KEYS),'Three exact feature matrices')
    mask=_bool(values['available'],shape,n,'availability');out={'available':mask}
    for key in ('candidate','control'):
        v=_numeric(values[key],shape,n,key,finite=False)
        require(np.isfinite(v[mask]).all() and np.isnan(v[~mask]).all(),'Common finite/NaN features')
        require((np.abs(v[mask])<=.5).all(),'Bounded target');out[key]=v
    return out


def decisions(study,arrays,meta,values,n):
    require(isinstance(arrays,dict) and isinstance(meta,dict),'Selection mappings')
    shape=np.shape(arrays['baseline']);require(len(shape)==2 and shape[1]>0 and type(n) is int and 0<=n<=shape[0],'Selection prefix')
    total,width=shape
    require(all(isinstance(meta.get(k),list) and len(meta[k])==total for k in ('dates','ends','symbols')),'Selection metadata axes')
    previous=None
    for i in range(n):
        day=axes._day(meta['dates'][i],'Selection date');end=axes._day(meta['ends'][i],'Selection end');names=meta['symbols'][i]
        require(end>day and (previous is None or day>previous),'Selection chronology');previous=day
        require(str(day.date())!=INITIAL_PARTIAL or i==0,'Named partial must be first')
        require(isinstance(names,list) and len(names)==width and all(type(x) is str and x and x.strip()==x for x in names) and len(set(names))==width,'Unique symbols')
    before=_snapshot(arrays);before_values=_snapshot(values);before_meta=deepcopy(meta);f=_features(values,shape,n)
    require(isinstance(meta.get('etfs'),(list,tuple)) and len(set(meta['etfs']))==len(meta['etfs']) and all(type(x) is str and x for x in meta['etfs']),'ETF identities')
    for i in range(n):
        require(not any(f['available'][i,j] for j,name in enumerate(meta['symbols'][i]) if name in meta['etfs']),'ETF targets must stay unavailable')
    base=_numeric(arrays['baseline'],shape,n,'baseline');core=_numeric(arrays['core'],shape,n,'weighted core');effect=_numeric(arrays['effects'],shape,n,'effects')
    good=_bool(arrays['eligible'],shape,n,'eligible');calm=_bool(arrays['state_calm10'],(total,),n,'calm');weak=_bool(arrays['state_weak_narrow'],(total,),n,'weak')
    rawcw=np.asarray(arrays['core_weight'])
    if rawcw.shape==(total,):cw=_numeric(rawcw,(total,),n,'core_weight')[:,None]
    else:
        require(rawcw.shape in ((total,1),shape),'Core weight shape');cw=_numeric(rawcw,rawcw.shape,n,'core_weight')
    cw=np.broadcast_to(cw,(n,width)).copy();require(np.array_equal(cw,np.broadcast_to(np.where(calm,.5,.52)[:,None],(n,width))),'Original core weights')
    with np.errstate(over='raise',invalid='raise',divide='raise'):support=(base-core)/(1.-cw)
    require(np.isfinite(support).all() and (np.abs(support)<=.5+1e-12).all(),'Original bounded support')
    extension=_bool(arrays['extension_available'],shape,n,'extension');forecast=_bool(arrays['forecast_available'],shape,n,'forecast')
    logrank=_numeric(arrays['rank_log_excess'],shape,n,'logrank',finite=False)
    ext=calm[:,None]&~weak[:,None]&extension&forecast&good&np.isfinite(logrank)
    active=np.array([meta['dates'][i]!=INITIAL_PARTIAL for i in range(n)],bool);use=active[:,None]&f['available']
    episode=np.minimum(.05,np.maximum(0.,.10-(.025*calm+.05*weak)))
    remaining=(1.-cw)-(.025*calm[:,None]+.05*weak[:,None]+.01*ext)-episode[:,None]
    funded=remaining.copy()
    require(np.isfinite(funded).all() and (funded>=.38-1e-12).all(),'Original support budget lower bound')
    pre=_numeric(study.modified_pre_score(study.configs()[4],arrays,n),shape,n,'original pre')
    episode_arrays=dict(baseline=base.copy(),core=core.copy(),core_weight=cw.copy(),eligible=good.copy(),state_calm10=calm.copy(),state_weak_narrow=weak.copy())
    scores={};pres={};orders={};deltas={};calls=[]
    for arm,key in zip(ARMS,(None,'candidate','control')):
        target=np.full((n,width),np.nan);delta=np.zeros((n,width))
        if key is not None:
            target[use]=f[key][use]
            with np.errstate(over='raise',invalid='raise'):delta[use]=BLEND*(cw[use]*target[use]-core[use])
        arm_pre=pre.copy();changed=delta!=0.
        with np.errstate(over='raise',invalid='raise'):arm_pre[changed]+=delta[changed]
        require(np.isfinite(delta).all() and np.isfinite(arm_pre).all(),'Finite half-core scores');arm_calls=[]
        def boundary(score,eligible,names,source_effects,original,policy):
            require(not arm_calls,'Exactly one episode per arm')
            require(np.array_equal(score,arm_pre) and np.array_equal(eligible,good),'Expected pre and original E')
            require(names==meta['symbols'][:n] and np.array_equal(source_effects,effect),'Original symbols/effects')
            for field,expected in (('baseline',base),('core',core),('core_weight',cw),('state_calm10',calm),('state_weak_narrow',weak)):
                require(np.array_equal(original[field],expected),'Original episode support '+field)
            require(policy==POLICY,'Original episode policy');arm_calls.append(True);calls.append(arm)
            return study.ep.select_boundary(score,eligible,names,source_effects,original,policy)
        chosen,post=core_adapter.score_and_choose('incumbent66',0. if key is None else BLEND,target,episode_arrays,meta,n,
            pre_score=pre,effects=effect,episode_policy=POLICY.copy(),boundary_selector=boundary)
        require(len(arm_calls)==1,'Exactly one original episode');post=_numeric(post,(n,width),n,'post score')
        require(isinstance(chosen,list) and len(chosen)==n,'Choice axes')
        scores[arm]=post.copy();pres[arm]=arm_pre.copy();orders[arm]=study.rp.make_orders(chosen,post,meta)
        if key is not None:deltas[key]=delta
    events=[]
    for i in range(n):
        event=dict(query_index=i,date=meta['dates'][i],initial_partial=meta['dates'][i]==INITIAL_PARTIAL,calm=bool(calm[i]),weak=bool(weak[i]),active=bool(active[i]),
            available_indices=np.flatnonzero(f['available'][i]).tolist(),extension_active_indices=np.flatnonzero(ext[i]).tolist(),episode_budget=float(episode[i]),
            candidate_delta=deltas['candidate'][i].tolist(),control_delta=deltas['control'][i].tolist(),original_remaining_support=remaining[i].tolist(),
            candidate_remaining_support=funded[i].tolist(),control_remaining_support=funded[i].tolist())
        require(tuple(event)==DECISION_EVENT_KEYS,'Decision event schema');events.append(event)
        if not active[i]:
            for arm in ARMS[1:]:require(orders[arm][i]==orders[ARMS[0]][i] and np.array_equal(scores[arm][i],scores[ARMS[0]][i]),'Initial exact incumbent')
    counts=dict(events=n,symbols_per_event=width,active_events=int(active.sum()),available_active_cells=int(use.sum()),
        candidate_nonzero_delta_cells=int(np.count_nonzero(deltas['candidate'])),control_nonzero_delta_cells=int(np.count_nonzero(deltas['control'])),
        original_episode_calls=len(calls),new_model_fits=0,predictive_model_fits=0)
    require(tuple(counts)==DECISION_COUNT_FIELDS,'Decision count schema')
    require(_snapshot(arrays)==before and _snapshot(values)==before_values and meta==before_meta,'Input mutation')
    return orders,scores,pres,dict(schema='monthly_residual_core246_decisions_v1',counts=counts,events=events)
