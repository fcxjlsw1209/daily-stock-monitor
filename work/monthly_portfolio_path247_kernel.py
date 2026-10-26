"""Prospective P-shortlist portfolio construction; caller supplies all data.

Only frozen code dependencies are imported. There are no loaders, model fits,
registrations, performance calculations or writes. The original episode runs
once per prepare/decisions call; portfolio utilities never replace original F.
"""
from copy import deepcopy
from datetime import date
import hashlib
import importlib.util
import itertools
import json
import math
from numbers import Real
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MATH_PATH=ROOT/'work/monthly_portfolio_path247_math.py'
MATH_SHA='8cb8162a9685b9ce207b312d42330065fef57e4831cae2c9891e95e764cf2966'
AXES_PATH=ROOT/'work/monthly_risk_consensus219_kernel.py'
AXES_SHA='7316b7cc081e7f8c25fc8cbeb51f43642223367609be1f77e4beb81929c1be77'
ADAPTER_PATH=ROOT/'outputs/monthly_path70_research/incumbent_adapter.py'
ADAPTER_SHA='5619bfc3512a0aa057e79042256a379360d6baa1bc0c448f2f301b84d6ec3e2b'


def _module(path,digest,name):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Changed frozen dependency: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


stats=_module(MATH_PATH,MATH_SHA,'portfolio247_frozen_math')
axes=_module(AXES_PATH,AXES_SHA,'portfolio247_axes219')
adapter=_module(ADAPTER_PATH,ADAPTER_SHA,'portfolio247_frozen_adapter')
ARMS=('incumbent_local67','portfolio_joint_path','portfolio_diagonal_control','portfolio_structure_control')
BUDGET=.025
CAP=.02
PRICE_COUNT=253
INITIAL_PARTIAL='2016-08-17'
POLICY=dict(budget=.05,ceiling=5,gap=.02)
COUNT_FIELDS=('events','symbols_per_event','eligible_cells','initial_partial_events','active_events','shortlist_cells',
 'source_available_cells','source_price_observations','enumerated_trios','feasible_trios','trio_statistics','zero_variance_trios',
 'missing_source_events','zero_variance_events','available_events','joint_changed_holdings','diagonal_changed_holdings',
 'structure_changed_holdings','original_episode_calls','new_model_fits','predictive_model_fits')
EVENT_KEYS=('query_index','date','symbols','initial_partial','calm','weak','core_weight','eligible_indices','pre_scores',
 'final_scores','weighted_core','incumbent_choices','pre_order','shortlist_indices','incumbent_trio','reference_raw_core_mean',
 'reference_final_mean','feasible_trio_indices','session_position','price_start','price_end','source_dates','clock_sessions',
 'source_cells','trios','branch')
TRIO_KEYS=('indices','symbols','raw_core_mean','weighted_core_mean','final_mean','auxiliary_mean','raw_core_regret',
 'final_regret','feasible','statistics')
SOURCE_KEYS=('column_index','symbol','source_available','reason','observations_checked','start_price','end_price','listing')
LISTING_KEYS=('issuer_key','not_before','available_from','evidence_id','visible','start_precedes_bound')
UTILITY_KEYS=('joint_q','diagonal_q','joint_utility','diagonal_utility','structural_utility')
BRANCHES=('initial_partial','fewer_than_three','missing_source','zero_variance','available')


def require(ok,message):
    if not ok:raise ValueError(message)


def dependency_files():
    return (MATH_PATH,AXES_PATH,ADAPTER_PATH,ROOT/'outputs/monthly_local_linear70/study.py',
        ROOT/'outputs/monthly_model_scope70/study.py',ROOT/'outputs/monthly_daily_episodes/episodes.py',
        ROOT/'outputs/monthly_return_priority/return_priority.py',ROOT/'outputs/monthly_local_linear70/INCUMBENT_POLICY.json')


def _iso(value,label):
    require(type(value) is str,label+' canonical ISO string')
    try:parsed=date.fromisoformat(value)
    except (ValueError,TypeError) as exc:raise ValueError(label+' canonical ISO date') from exc
    require(parsed.isoformat()==value,label+' canonical ISO date');return value


def _real(value,label):
    require(isinstance(value,Real) and not isinstance(value,(bool,np.bool_)),label+' real nonboolean')
    try:x=float(value)
    except (OverflowError,ValueError,TypeError) as exc:raise ValueError(label+' binary64') from exc
    require(math.isfinite(x),label+' finite');return x


def _numeric(value,shape,n,label):
    raw=np.asarray(value);require(raw.shape==shape,label+' exact shape')
    out=np.empty(raw[:n].shape,float)
    for ix in np.ndindex(out.shape):out[ix]=_real(raw[ix],label)
    return out


def _bool(value,shape,n,label):
    raw=np.asarray(value);require(raw.shape==shape and raw.dtype.kind=='b',label+' bool shape');return raw[:n].copy()


def _snapshot(arrays):
    return {key:(np.asarray(value).shape,np.asarray(value).dtype.str,np.asarray(value).tobytes()) for key,value in arrays.items()}


def _bridge(study,arrays,meta,n):
    require(isinstance(arrays,dict) and isinstance(meta,dict),'Input mappings')
    shape=np.shape(arrays['baseline']);require(len(shape)==2 and shape[1]>0 and type(n) is int and 0<=n<=shape[0],'Nonempty-width prefix')
    total,width=shape
    require(all(isinstance(meta.get(k),list) and len(meta[k])==total for k in ('dates','ends','symbols')),'Metadata axes')
    previous=None
    for i in range(n):
        day=_iso(meta['dates'][i],'query');end=_iso(meta['ends'][i],'end');names=meta['symbols'][i]
        require(end>day and (previous is None or day>previous),'Increasing dates/end');previous=day
        require(day!=INITIAL_PARTIAL or i==0,'Named initial partial first')
        require(isinstance(names,list) and len(names)==width and all(type(s) is str and s and s.strip()==s for s in names)
                and len(set(names))==width,'Unique exact symbols')
    base=_numeric(arrays['baseline'],shape,n,'baseline');core=_numeric(arrays['core'],shape,n,'core')
    effect=_numeric(arrays['effects'],shape,n,'effects');eligible=_bool(arrays['eligible'],shape,n,'eligible')
    calm=_bool(arrays['state_calm10'],(total,),n,'calm');weak=_bool(arrays['state_weak_narrow'],(total,),n,'weak')
    rawcw=np.asarray(arrays['core_weight'])
    require(rawcw.shape in ((total,),(total,1),shape),'Core weight shape')
    cw=_numeric(rawcw,rawcw.shape,n,'core weight')
    if cw.ndim==1:cw=cw[:,None]
    cw=np.broadcast_to(cw,(n,width)).copy()
    require(np.array_equal(cw,np.broadcast_to(np.where(calm,.5,.52)[:,None],(n,width))),'Original common .50/.52 weights')
    support=(base-core)/(1.-cw)
    require(np.isfinite(support).all() and (np.abs(support)<=.5+1e-12).all(),'Original bounded support')
    # Physically copy only the consumed rows before calling legacy helpers, whose
    # core-weight conversion otherwise scans future values even for logical n.
    call_arrays={}
    for key,value in arrays.items():
        raw=np.asarray(value);require(raw.ndim>=1 and raw.shape[0]==total,'Cached monthly axis '+key)
        call_arrays[key]=raw[:n].copy()
    call_arrays['core_weight']=cw.copy()
    call_meta=dict(meta,dates=meta['dates'][:n],ends=meta['ends'][:n],symbols=meta['symbols'][:n])
    pre=_numeric(study.modified_pre_score(study.configs()[4],call_arrays,n),(n,width),n,'original pre')
    called=[]
    def boundary(score,good,names,effects,original,policy):
        require(not called,'Only one original episode')
        require(np.array_equal(score,pre) and np.array_equal(good,eligible),'Original pre/E')
        require(names==meta['symbols'][:n] and np.array_equal(effects,effect),'Original identities/effects')
        for key,expected in (('baseline',base),('core',core),('core_weight',cw),('state_calm10',calm),('state_weak_narrow',weak)):
            require(np.array_equal(original[key],expected),'Original episode '+key)
        require(policy==POLICY,'Original episode policy');called.append(True)
        return study.ep.select_boundary(score,good,names,effects,original,policy)
    chosen,post=adapter.score_and_choose('incumbent66',0.,np.full((n,width),np.nan),call_arrays,call_meta,n,
        pre_score=pre,effects=effect,episode_policy=POLICY.copy(),boundary_selector=boundary)
    require(len(called)==1,'One original episode required')
    post=_numeric(post,(n,width),n,'original final');require(isinstance(chosen,list) and len(chosen)==n,'Original choice axis')
    for i in range(n):
        names=meta['symbols'][i];rank=sorted(np.flatnonzero(eligible[i]).tolist(),key=lambda j:(-pre[i,j],names[j]))
        if len(rank)<3:expected=rank if len(rank)==2 else []
        else:
            js=[j for j in rank[2:5] if pre[i,rank[2]]-pre[i,j]<=.02]
            win=min(js,key=lambda j:(-post[i,j],names[j]));expected=sorted(rank[:2]+[win],key=lambda j:(-post[i,j],names[j]))
        require(chosen[i]==expected,'Original frozen boundary choice')
    return dict(width=width,pre=pre,post=post,core=core,cw=cw,eligible=eligible,calm=calm,weak=weak,
                choices=[[int(j) for j in row] for row in chosen])


def _mean(values,label):
    return stats._divide(stats._sum(values,label),3.,label)


def _trio(indices,names,bridge,i):
    ids=sorted(indices,key=lambda j:names[j]);core=bridge['core'][i];post=bridge['post'][i];cw=float(bridge['cw'][i,0])
    return dict(indices=ids,symbols=[names[j] for j in ids],
        raw_core_mean=_mean([stats._divide(float(core[j]),cw,'raw core') for j in ids],'mean raw core'),
        weighted_core_mean=_mean([float(core[j]) for j in ids],'mean weighted core'),
        final_mean=_mean([float(post[j]) for j in ids],'mean final'),
        auxiliary_mean=_mean([stats._difference(float(post[j]),float(core[j]),'final auxiliary') for j in ids],'mean auxiliary'),
        raw_core_regret=None,final_regret=None,feasible=None,statistics=None)


def _domain(meta,bridge,i):
    names=meta['symbols'][i];partial=meta['dates'][i]==INITIAL_PARTIAL
    eligible=np.flatnonzero(bridge['eligible'][i]).tolist()
    rank=sorted(eligible,key=lambda j:(-bridge['pre'][i,j],names[j]));inc=bridge['choices'][i]
    active=not partial and len(rank)>=3;shortlist=rank[:5] if active else []
    reference=_trio(inc,names,bridge,i) if active else None
    trios=[]
    if active:
        for ids in itertools.combinations(sorted(shortlist,key=lambda j:names[j]),3):
            item=_trio(ids,names,bridge,i)
            item['raw_core_regret']=stats._difference(reference['raw_core_mean'],item['raw_core_mean'],'raw-core regret')
            item['final_regret']=stats._difference(reference['final_mean'],item['final_mean'],'final regret')
            item['feasible']=(item['raw_core_mean']>=reference['raw_core_mean']-CAP and item['final_mean']>=reference['final_mean']-CAP)
            trios.append(item)
        require(any(z['indices']==reference['indices'] and z['feasible'] for z in trios),'Incumbent retained in feasible P shortlist')
    return dict(query_index=i,date=meta['dates'][i],symbols=names.copy(),initial_partial=partial,
        calm=bool(bridge['calm'][i]),weak=bool(bridge['weak'][i]),core_weight=float(bridge['cw'][i,0]),
        eligible_indices=eligible,pre_scores=bridge['pre'][i].tolist(),final_scores=bridge['post'][i].tolist(),
        weighted_core=bridge['core'][i].tolist(),incumbent_choices=inc.copy(),pre_order=rank,shortlist_indices=shortlist,
        incumbent_trio=reference['indices'] if reference else None,
        reference_raw_core_mean=reference['raw_core_mean'] if reference else None,
        reference_final_mean=reference['final_mean'] if reference else None,
        feasible_trio_indices=[k for k,z in enumerate(trios) if z['feasible']],
        session_position=None,price_start=None,price_end=None,source_dates=[],clock_sessions=[],source_cells=[],
        trios=trios,branch='initial_partial' if partial else ('fewer_than_three' if not active else None))


def _window_record(day,t,sessions,i):
    require(t>=294 and t+1<len(sessions),'Complete official formation and next-session witness')
    ds=sessions[t-294:t-41].strftime('%Y-%m-%d').tolist()
    return dict(query_index=i,date=day,initial_partial=day==INITIAL_PARTIAL,query_session_position=t,
        price_start=ds[0],first_return_date=ds[1],price_end=ds[-1],source_start_position=t-294,source_end_position=t-42,
        prices=253,return_intervals=252,next_official_session=str(sessions[t+1].date()),
        source_dates_sha256=hashlib.sha256(json.dumps(ds,separators=(',',':')).encode()).hexdigest())


def _indexed(records,n,label):
    require(isinstance(records,list),label+' list');out={}
    for record in records:
        require(isinstance(record,dict) and type(record.get('query_index')) is int and record['query_index']>=0,label+' query index')
        i=record['query_index']
        if i>=n:continue
        require(i not in out,label+' duplicate query');out[i]=record
    require(set(out)==set(range(n)),label+' exact consumed queries');return out


def _source_setup(panel,meta,calendar,review,n):
    require(isinstance(panel,dict) and set(panel)=={'adj'},'Exactly original adjusted panel')
    axes._frame(panel['adj']);require(isinstance(review,dict),'Source metadata review')
    sessions=axes._calendar(calendar) if len(calendar) else pd.DatetimeIndex([])
    queries,_=axes._queries(dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n]),sessions)
    require(isinstance(review.get('calendar'),dict),'Review calendar')
    windows=_indexed(review['calendar'].get('windows'),n,'Review windows')
    bounds=review.get('listing_bounds');cells=review.get('listing_identity_cells')
    require(isinstance(bounds,list) and isinstance(cells,list),'Listing evidence lists')
    bysymbol={}
    for b in bounds:
        require(isinstance(b,dict) and all(type(b.get(k)) is str and b[k] for k in ('symbol','issuer_key','not_before','available_from','evidence_id','kind')),'Listing evidence fields')
        _iso(b['not_before'],'not before');_iso(b['available_from'],'available from')
        require(b['symbol'] not in bysymbol,'Unique listing symbols');bysymbol[b['symbol']]=b
    seen={}
    for c in cells:
        require(isinstance(c,dict) and type(c.get('query_index')) is int and c['query_index']>=0,'Listing query index')
        i=c['query_index']
        if i>=n:continue
        j=c.get('column_index');require(type(j) is int and 0<=j<len(meta['symbols'][i]),'Listing column')
        require((i,j) not in seen,'Duplicate listing identity');seen[i,j]=c
    expected={(i,j) for i in range(n) for j,symbol in enumerate(meta['symbols'][i]) if symbol in bysymbol}
    require(set(seen)==expected,'Complete consumed reviewed issuer occurrences')
    for i,(day,names,t) in enumerate(queries):
        expected_window=_window_record(str(day.date()),t,sessions,i)
        require(all(windows[i].get(k)==v for k,v in expected_window.items()),'Review official window mismatch')
        for j,symbol in enumerate(names):
            if symbol not in bysymbol:continue
            c=seen[i,j];b=bysymbol[symbol]
            exact=dict(date=str(day.date()),symbol=symbol,issuer_key=b['issuer_key'],new_price_start=expected_window['price_start'],
                new_price_end=expected_window['price_end'],bound_visible=b['available_from']<=str(day.date()),
                start_precedes_bound=expected_window['price_start']<b['not_before'])
            require(all(c.get(k)==v for k,v in exact.items()),'Reviewed listing identity/window mismatch')
    return panel['adj'],sessions,queries,bysymbol


def _extract(adj,dates,j,symbol,bound,day):
    listing=None
    if bound:
        listing=dict(issuer_key=bound['issuer_key'],not_before=bound['not_before'],available_from=bound['available_from'],
            evidence_id=bound['evidence_id'],visible=bound['available_from']<=day,start_precedes_bound=str(dates[0].date())<bound['not_before'])
    source=dict(column_index=j,symbol=symbol,source_available=False,reason=None,observations_checked=0,
                start_price=None,end_price=None,listing=listing)
    if symbol not in adj.columns:source['reason']='missing_symbol';return source,None
    missing=dates.difference(adj.index);present=dates.intersection(adj.index);values=[];nan=False
    for raw in adj.loc[present,symbol].to_numpy(copy=True):
        require(isinstance(raw,Real) and not isinstance(raw,(bool,np.bool_)),'Malformed required price type')
        try:value=float(raw)
        except (ValueError,TypeError,OverflowError) as exc:raise ValueError('Malformed required binary64 price') from exc
        if math.isnan(value):nan=True
        else:require(math.isfinite(value) and value>0.,'Malformed required positive finite price')
        values.append(value)
    source['observations_checked']=len(values)
    source['reason']='missing_official_session' if len(missing) else ('missing_nan' if nan else 'available')
    source['source_available']=source['reason']=='available'
    if source['source_available']:
        require(len(values)==PRICE_COUNT,'Complete253 price count')
        source['start_price']=values[0];source['end_price']=values[-1]
        require(not(listing and listing['visible'] and listing['start_precedes_bound']),'Complete source contradicts visible listing identity')
    return source,values if source['source_available'] else None


def _statistic_check(z):
    require(isinstance(z,dict) and z.get('schema')==stats.SCHEMA and z.get('price_count')==253 and z.get('return_count')==252,'Cached trio statistic identity')
    for key,size,rows in (('normalized_prices',253,3),('contributions',252,3)):
        require(isinstance(z.get(key),list) and len(z[key])==rows and all(isinstance(x,list) and len(x)==size for x in z[key]),'Cached statistic series shape '+key)
    require(isinstance(z.get('portfolio_returns'),list) and len(z['portfolio_returns'])==252,'Cached return series shape')
    mean=_real(z.get('mean_return'),'cached mean');joint=_real(z.get('joint_variance'),'cached joint variance');diag=_real(z.get('diagonal_variance'),'cached diagonal variance')
    require(joint>=0 and diag>=0,'Cached nonnegative variances');good=joint>0 and diag>0
    require(type(z.get('available')) is bool and z['available']==good and z.get('reason')==('available' if good else 'zero_variance'),'Cached exact variance gate')
    require(z.get('joint_sd')==math.sqrt(joint) and z.get('diagonal_sd')==math.sqrt(diag),'Cached exact SD')
    if good:
        require(z.get('joint_ratio')==stats._divide(mean,math.sqrt(joint),'cached joint ratio') and
                z.get('diagonal_ratio')==stats._divide(mean,math.sqrt(diag),'cached diagonal ratio'),'Cached canonical ratios')
    else:require(z.get('joint_ratio') is None and z.get('diagonal_ratio') is None,'Both zero-support ratios absent')
    return good


def _cached_event(feature,expected):
    require(isinstance(feature,dict) and set(feature)==set(EVENT_KEYS),'Cached feature schema')
    variable={'session_position','price_start','price_end','source_dates','clock_sessions','source_cells','trios','branch'}
    for key in EVENT_KEYS:
        if key not in variable:require(feature[key]==expected[key],'Cached original binding '+key)
    require(type(feature['session_position']) is int and feature['session_position']>=294,'Cached session position')
    clock=feature['clock_sessions'];ds=feature['source_dates']
    require(isinstance(clock,list) and len(clock)==296 and all(type(v) is str for v in clock),'Cached complete clock')
    for d in clock:_iso(d,'cached clock')
    require(clock==sorted(set(clock)) and all(date.fromisoformat(d).weekday()<5 for d in clock),'Cached unique increasing weekday clock')
    require(ds==clock[:253] and feature['price_start']==ds[0] and feature['price_end']==ds[-1] and clock[294]==feature['date'],'Cached exact253/query clock')
    require(feature['initial_partial'] or clock[-1][:7]!=feature['date'][:7],'Cached official month-end witness')
    require(isinstance(feature['trios'],list) and len(feature['trios'])==len(expected['trios']),'Cached trio inventory')
    for actual,wanted in zip(feature['trios'],expected['trios']):
        require(isinstance(actual,dict) and set(actual)==set(TRIO_KEYS),'Cached trio schema')
        for key in TRIO_KEYS:
            if key!='statistics':require(actual[key]==wanted[key],'Cached trio domain '+key)
    active=bool(expected['shortlist_indices']);cells=feature['source_cells']
    require(isinstance(cells,list) and len(cells)==len(expected['shortlist_indices']),'Cached shortlist source inventory')
    for c,j in zip(cells,expected['shortlist_indices']):
        require(isinstance(c,dict) and set(c)==set(SOURCE_KEYS) and c['column_index']==j and c['symbol']==feature['symbols'][j],'Cached source identity')
        require(type(c['source_available']) is bool and c['reason'] in ('available','missing_symbol','missing_official_session','missing_nan'),'Cached source reason')
        require(c['source_available']==(c['reason']=='available'),'Cached source gate')
        require(type(c['observations_checked']) is int and 0<=c['observations_checked']<=253,'Cached observations')
        if c['source_available']:
            require(c['observations_checked']==253 and _real(c['start_price'],'cached first price')>0 and _real(c['end_price'],'cached last price')>0,'Cached complete source')
        else:require(c['start_price'] is None and c['end_price'] is None,'Unavailable endpoints absent')
        listing=c['listing']
        if listing is not None:
            require(isinstance(listing,dict) and set(listing)==set(LISTING_KEYS),'Cached listing schema')
            _iso(listing['not_before'],'cached listing bound');_iso(listing['available_from'],'cached listing availability')
            require(type(listing['issuer_key']) is str and listing['issuer_key'] and type(listing['evidence_id']) is str and listing['evidence_id'],'Cached listing evidence')
            require(listing['visible']==(listing['available_from']<=feature['date']) and listing['start_precedes_bound']==(feature['price_start']<listing['not_before']),'Cached listing time')
            require(not(c['source_available'] and listing['visible'] and listing['start_precedes_bound']),'Cached listing contradiction')
    complete=active and all(c['source_available'] for c in cells)
    good=[]
    for actual in feature['trios']:
        should=complete and actual['feasible']
        if should:good.append(_statistic_check(actual['statistics']))
        else:require(actual['statistics'] is None,'No statistics outside common feasible source')
    branch=expected['branch'] if not active else ('missing_source' if not complete else ('available' if all(good) else 'zero_variance'))
    require(feature['branch']==branch,'Cached common branch')
    return deepcopy(feature)


def _rank(values):
    total=len(values);result=[]
    for value in values:
        less=sum(v<value for v in values);equal=sum(v==value for v in values)
        result.append((float(less)+float(equal+1)/2.)/float(total)-.5)
    return result


def _select(study,meta,bridge,features,n):
    require(isinstance(features,list) and len(features)>=n,'Feature prefix list')
    width=bridge['width'];counts=dict.fromkeys(COUNT_FIELDS,0)
    counts.update(events=n,symbols_per_event=width,original_episode_calls=1)
    scores={arm:bridge['post'].copy() for arm in ARMS};pres={arm:bridge['pre'].copy() for arm in ARMS}
    choices={arm:[] for arm in ARMS};history=[]
    for i in range(n):
        event=_cached_event(features[i],_domain(meta,bridge,i));inc=event['incumbent_choices'];picked={arm:inc.copy() for arm in ARMS}
        scale=stats._difference(1.,stats._divide(BUDGET,1.-event['core_weight'],'funding share'),'noncore scale')
        require(0.<scale<1. and event['core_weight']>=.5,'Funded core coefficients')
        for t in event['trios']:
            for key in UTILITY_KEYS:t[key]=None
        if event['branch']=='available':
            hs=event['feasible_trio_indices']
            for field,raw in (('joint_q','joint_ratio'),('diagonal_q','diagonal_ratio')):
                qs=_rank([event['trios'][k]['statistics'][raw] for k in hs])
                for k,q in zip(hs,qs):event['trios'][k][field]=q
            for k in hs:
                t=event['trios'][k]
                for prefix in ('joint','diagonal'):
                    scaled=stats._product(scale,t['auxiliary_mean'],'scaled final auxiliary')
                    first=stats._finite(t['weighted_core_mean']+scaled,'utility core plus auxiliary')
                    extra=stats._product(BUDGET,t[prefix+'_q'],'funded quality')
                    t[prefix+'_utility']=stats._finite(first+extra,'basket utility')
                t['structural_utility']=t['final_mean']
            for arm,key in zip(ARMS[1:],('joint_utility','diagonal_utility','structural_utility')):
                best=min(hs,key=lambda k:(-event['trios'][k][key],event['trios'][k]['indices']!=event['incumbent_trio'],tuple(event['trios'][k]['symbols'])))
                selected=event['trios'][best]['indices']
                picked[arm]=inc.copy() if set(selected)==set(inc) else sorted(selected,key=lambda j:(-bridge['post'][i,j],meta['symbols'][i][j]))
        event['noncore_scale']=scale;event['choices']=picked
        for arm in ARMS:choices[arm].append(picked[arm])
        counts['eligible_cells']+=len(event['eligible_indices']);counts['initial_partial_events']+=int(event['initial_partial'])
        counts['active_events']+=int(bool(event['shortlist_indices']));counts['shortlist_cells']+=len(event['source_cells'])
        counts['source_available_cells']+=sum(c['source_available'] for c in event['source_cells'])
        counts['source_price_observations']+=sum(c['observations_checked'] for c in event['source_cells'])
        counts['enumerated_trios']+=len(event['trios']);counts['feasible_trios']+=len(event['feasible_trio_indices'])
        measured=[t['statistics'] for t in event['trios'] if t['statistics'] is not None]
        counts['trio_statistics']+=len(measured);counts['zero_variance_trios']+=sum(not z['available'] for z in measured)
        for branch in ('missing_source','zero_variance','available'):counts[branch+'_events']+=int(event['branch']==branch)
        for arm,key in zip(ARMS[1:],('joint_changed_holdings','diagonal_changed_holdings','structure_changed_holdings')):
            counts[key]+=int(set(picked[arm])!=set(inc))
        history.append(event)
    orders={arm:study.rp.make_orders(choices[arm],scores[arm],meta) for arm in ARMS}
    return orders,scores,pres,dict(schema='monthly_portfolio_path247_decisions_v1',counts=counts,events=history)


def prepare(study,arrays,meta,panel,calendar,review,n):
    before=_snapshot(arrays);before_meta=deepcopy(meta)
    bridge=_bridge(study,arrays,meta,n)
    adj,sessions,queries,bounds=_source_setup(panel,meta,calendar,review,n);features=[]
    for i,(day,names,t) in enumerate(queries):
        event=_domain(meta,bridge,i);dates=sessions[t-294:t-41]
        event.update(session_position=t,price_start=str(dates[0].date()),price_end=str(dates[-1].date()),
                     source_dates=dates.strftime('%Y-%m-%d').tolist(),clock_sessions=sessions[t-294:t+2].strftime('%Y-%m-%d').tolist())
        if event['shortlist_indices']:
            rows={}
            for j in event['shortlist_indices']:
                cell,prices=_extract(adj,dates,j,names[j],bounds.get(names[j]),str(day.date()))
                event['source_cells'].append(cell);rows[j]=prices
            if not all(c['source_available'] for c in event['source_cells']):event['branch']='missing_source'
            else:
                for k in event['feasible_trio_indices']:
                    trio=event['trios'][k];trio['statistics']=stats.measure([rows[j] for j in trio['indices']])
                event['branch']='available' if all(event['trios'][k]['statistics']['available'] for k in event['feasible_trio_indices']) else 'zero_variance'
        require(tuple(event)==EVENT_KEYS,'Feature event schema');features.append(event)
    orders,scores,pres,audit=_select(study,meta,bridge,features,n)
    require(_snapshot(arrays)==before and meta==before_meta,'Original array/metadata mutation')
    return features,orders,scores,pres,audit


def decisions(study,arrays,meta,features,n):
    before=_snapshot(arrays);before_meta=deepcopy(meta)
    bridge=_bridge(study,arrays,meta,n)
    out=_select(study,meta,bridge,features,n)
    require(_snapshot(arrays)==before and meta==before_meta,'Original array/metadata mutation')
    return out
