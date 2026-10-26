"""263 independent full-history domain, canonical risk-clock and selection proofs.

Only SHA-pinned older independent helpers are reused. No263 producer/host import,
raw-data loader, predictive fitting or lifecycle entry point exists here.
"""
import copy
import hashlib
import importlib.util
import math
from numbers import Real
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'work/monthly_peer_path252_independent_v2.py'
PRIOR_SHA='02d5d53a4f4867057f58f8cd6295b7838a34b75d6fbc477e6a49016a9a3a282b'
ARMS=('incumbent_local67','canonical_current63_core_control','canonical_lag63_core_candidate')
INITIAL='2016-08-17'
RISK_KEYS=('mean','centered','squares','square_sum','variance','sd','constant')
C_SOURCE_ATOL=1e-12
C_SOURCE_RTOL=0.

def load_pinned(path,digest,name):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Frozen independent dependency changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

old=load_pinned(PRIOR,PRIOR_SHA,'risk_clock263_frozen_independent')
peer=old.peer
core=old.old
require,record_check,exact_array=peer.require,peer.record_check,peer.exact_array
PATHS,ACCOUNTING_SHA=old.PATHS,old.ACCOUNTING_SHA

def accounting():return old.accounting()
def dependency_files():return tuple(sorted(set(old.dependency_files())|{PRIOR,Path(__file__).resolve()}))
def _calendar(x):return x['sessions'] if type(x) is dict else x

def finite(x,label='number'):
    require(isinstance(x,Real) and not isinstance(x,(bool,np.bool_)),'Real nonboolean '+label)
    try:v=float(x)
    except (ValueError,TypeError,OverflowError) as exc:raise ValueError('Unrepresentable '+label) from exc
    require(math.isfinite(v),'Finite '+label)
    if x!=0 and v==0.:raise ArithmeticError('Nonzero conversion underflow '+label)
    return v

def quotient(a,b):
    z=float(a/b)
    if not math.isfinite(z) or a!=0. and z==0.:raise ArithmeticError('Division overflow/underflow')
    return z

def risk63(values):
    require(len(values)==63,'Exactly63 risk returns')
    values=[finite(x,'return') for x in values]
    constant=all(x==values[0] for x in values)
    if constant:return dict(mean=values[0],centered=[0.]*63,squares=[0.]*63,square_sum=0.,variance=0.,sd=0.,constant=True)
    mean=quotient(math.fsum(values),63.);centered=[];squares=[]
    for x in values:
        residual=finite(x-mean,'centered return');square=float(residual*residual)
        if not math.isfinite(square) or residual!=0. and square==0.:raise ArithmeticError('Square overflow/underflow')
        centered.append(residual);squares.append(square)
    total=finite(math.fsum(squares),'square sum');variance=quotient(total,62.);sd=finite(math.sqrt(variance),'sample SD')
    require(total>0. and variance>0. and sd>0.,'Nonconstant risk strictly positive')
    return dict(mean=mean,centered=centered,squares=squares,square_sum=total,variance=variance,sd=sd,constant=False)

def ranks(values):
    xs=[finite(v,'rank input') for v in values];N=len(xs)
    return [(float(2*sum(y<x for y in xs)+sum(y==x for y in xs)+1)/2.)/float(N)-.5 for x in xs]

def _n(meta,n):
    if n is None:n=len(meta['dates'])
    require(type(n) is int and 0<=n<=len(meta['dates']),'Native consumed prefix')
    return n

def original_evidence(panel,meta,calendar,arrays,n=None):
    n=_n(meta,n);base=core.original_base(arrays,meta,n)
    frame,sessions=core.source_axes(panel,_calendar(calendar),n)
    positions={d:i for i,d in enumerate(sessions)};events=[];observations=checked=finite_cells=0;maximum=0.
    for q,day in enumerate(base['days']):
        require(day in positions,'Original official query');t=positions[day]
        require(t>=294 and t+1<len(sessions),'Original history and next-session witness')
        partial=meta['dates'][q]==INITIAL
        if partial:require(q==0,'Named partial at first query')
        else:require(sessions[t+1].to_period('M')!=day.to_period('M'),'Original official monthend')
        e=dict(query_index=q,date=str(day.date()),initial_partial=partial,calm=bool(base['calm'][q]),u0_indices=[],original_raw=None,original_rank=None,frozen_core=None,rebuilt_core=None,core_abs_error=None,original_sigma63=None)
        if not partial:
            names=base['names'][q];require(all(s in frame.columns for s in names),'Original full inventory columns')
            past=frame.loc[:day]
            require(len(past)>=295 and sessions[0]==pd.Timestamp('2008-01-02') and past.index.equals(pd.DatetimeIndex(sessions[:t+1])),'Original full official2008 clock')
            raw,sigma=core.original_values(frame,day,names,'2008-01-02')
            chosen=[float(row[0 if base['calm'][q] else 1]) for row in raw]
            sanitized=[x if math.isfinite(x) else math.nan for x in chosen]
            ranked,U=core.independent_rank(sanitized);rebuilt=base['cw'][q]*ranked;errors=np.abs(rebuilt-base['core'][q])
            require(np.isfinite(errors).all() and (errors<=C_SOURCE_ATOL).all(),'Original weighted core source parity')
            e.update(u0_indices=U,original_raw=[v if math.isfinite(v) else None for v in chosen],original_rank=ranked.tolist(),frozen_core=base['core'][q].tolist(),rebuilt_core=rebuilt.tolist(),core_abs_error=errors.tolist(),original_sigma63=[v if math.isfinite(v) else None for v in sigma])
            observations+=len(past)*len(names);checked+=len(names);finite_cells+=len(U);maximum=max(maximum,float(errors.max()))
        events.append(e)
    return base,frame,sessions,events,dict(original_source_price_observations=observations,original_source_cells=checked,original_checked_cells=checked,original_finite_cells=finite_cells,maximum_core_source_absolute_error=maximum)

SOURCE_SCHEMA='risk_clock263_metadata_source_contract_v1'
EVENT_KEYS=('query_index','date','symbols','initial_partial','calm','weak','core_weight','query_position','next_official_session','clock_sessions','original_price_origin','u0_indices','active_indices','numerator_positions','numerator_dates','current_positions','current_dates','lag_positions','lag_dates','required_positions','required_dates','cells','zero_current_indices','zero_lag_indices','branch','current_target','lag_target')
CELL_KEYS=('column_index','symbol','original_finite','original_raw','original_rank','frozen_core','rebuilt_core','core_abs_error','original_sigma63','history_bound','observations','statistics')
STAT_KEYS=('schema','momentum','current_returns','lag_returns','current_risk','lag_risk','current_ratio','lag_ratio')
COUNT_FIELDS=('events','symbols_per_event','eligible_cells','initial_partial_events','calm_events','active_events','original_core_cells','original_finite_cells','required_cells','price_observations','current_return_observations','lag_return_observations','current_risk_statistics','lag_risk_statistics','zero_current_cells','zero_lag_cells','zero_risk_fallback_events','empty_domain_events','available_events','ratio_evaluations','ranked_cells','current_nonzero_deltas','lag_nonzero_deltas','current_changed_holdings','lag_changed_holdings','candidate_control_changed_holdings','original_episode_calls','new_model_fits','predictive_model_fits')
DECISION_EVENT_KEYS=('query_index','date','initial_partial','calm','weak','core_weight','branch','eligible_indices','u0_indices','original_support','current_deltas','lag_deltas','choices','pre_order')
ORIGINAL_EVENT_KEYS=('query_index','date','initial_partial','calm','u0_indices','original_raw','original_rank','frozen_core','rebuilt_core','core_abs_error','original_sigma63')
FEATURE_FLAGS=('passed','independent_source_evidence_checked','original_full_history_domain_checked','all_U0_before_eligibility','original_etf_domain_preserved','official107_price_clock_checked','visible_history_bounds_checked','required_source_hard_failures_checked','canonical_numerator_checked','canonical_risk_statistics_checked','whole_query_zero_policy_checked','signed_momentum_preserved','no_future_or_outcome_inputs','descriptive_counts_checked','exact_discrete')
ORIGINAL_SOURCE_FLAGS=tuple(dict.fromkeys(old.ORIGINAL_SOURCE_FLAGS+('original_full_history_grid_checked','original_U0_evidence_supplied')))
INCUMBENT_FLAGS=('passed','full_scores_supplied','full_pre_scores_supplied','independent_original_pre_checked','exact_incumbent_copy_checked','original_eligibility_unchanged','single_original_episode')
INCUMBENT_GATE_FLAGS=INCUMBENT_FLAGS
SELECTION_FLAGS=('passed','independent_original_pre_checked','original_eligibility_unchanged','original_support_checked','original_core_support_preserved','full_core_delta_checked','frozen_core_delta_anchor_checked','shared_rank_domain_checked','canonical_percentile_ties_checked','whole_query_zero_policy_checked','original_budget_checked','calm_nonweak_extension_budget_checked','original_effects_unchanged','one_episode_per_arm','no_final_global_rerank','initial_partial_unchanged','calm_unchanged','full_audit_checked','exact_scores_and_choices_checked','exact_discrete')
PREPARE_FLAGS=tuple(dict.fromkeys(FEATURE_FLAGS+SELECTION_FLAGS+('independent_source_rebuild',)))
WORK_FIELDS=('full_source_slot_inspections','cached_slot_validations','momentum_evaluations','return_evaluations','statistic_evaluations','ratio_evaluations','rank_evaluations','scalar_rank_evaluations','original_episode_calls','original_source_price_observations','original_source_cells')

def work_receipt(n,*,slots=0,required=0,ratios_count=0,domains=0,scalars=0,episodes=0,rebuilt=False,original=None):
    result=dict.fromkeys(WORK_FIELDS,0)
    result.update(checked_events=n,producer_modules_imported=False,computes_returns=False,full_source_slot_inspections=slots if rebuilt else 0,cached_slot_validations=0 if rebuilt else slots,momentum_evaluations=required,return_evaluations=126*required,statistic_evaluations=2*required,ratio_evaluations=ratios_count,rank_evaluations=domains,scalar_rank_evaluations=scalars,original_episode_calls=episodes)
    if original:
        for key in ('original_source_price_observations','original_source_cells'):result[key]=original[key]
    return result

def verify_original_source(panel,meta,calendar,arrays,n=None):
    base,frame,sessions,events,counts=original_evidence(panel,meta,calendar,arrays,n);n=len(events)
    return dict(**{k:True for k in ORIGINAL_SOURCE_FLAGS},**work_receipt(n,original=counts),**{k:v for k,v in counts.items() if k not in WORK_FIELDS},original_events=events,u0_indices=[e['u0_indices'] for e in events],C_source_atol=C_SOURCE_ATOL,C_source_rtol=C_SOURCE_RTOL,numerical_tolerance=0.,rank_tolerance=0.,episode_calls=0,new_target_ranks_computed=False,candidate_selection_computed=False)

def measure(prices107):
    require(len(prices107)==107,'Exactly107 distinct prices')
    p=[finite(x,'required price') for x in prices107];require(all(x>0. for x in p),'Positive required prices')
    momentum=finite(quotient(p[64],p[0])-1.,'Direct momentum')
    def ret(first,last):
        out=[]
        for k in range(first+1,last+1):
            gross=quotient(p[k],p[k-1]);require(gross>0.,'Positive simple gross');out.append(finite(gross-1.,'Simple return'))
        return out
    current=ret(43,106);lag=ret(1,64)
    return dict(schema='monthly_risk_clock263_statistics_v1',momentum=momentum,current_returns=current,lag_returns=lag,current_risk=risk63(current),lag_risk=risk63(lag),current_ratio=None,lag_ratio=None)

def _bound(symbol,day,partial=False):
    item=peer.BOUNDS.get(symbol)
    return None if partial or item is None or day<item['available_from'] else copy.deepcopy(item)

def _indices(values,width,label):
    require(type(values) is list and all(type(j) is int and 0<=j<width for j in values) and values==sorted(set(values)),label)

def event_clock(q,base,sessions):
    day=base['days'][q];t=sessions.index(day);d=str(day.date());partial=d==INITIAL
    require(t>=294 and t+1<len(sessions),'Official294 history/witness')
    if partial:require(q==0,'Initial partial position')
    else:require(sessions[t+1].to_period('M')!=day.to_period('M'),'Official monthend')
    positions=[] if partial else [t-294]+list(range(t-105,t+1))
    numerator=[] if partial else [t-294,t-42];current=[] if partial else list(range(t-63,t+1));lag=[] if partial else list(range(t-105,t-41))
    dates=lambda ids:[str(sessions[k].date()) for k in ids]
    return dict(query_index=q,date=d,symbols=list(base['names'][q]),initial_partial=partial,calm=bool(base['calm'][q]),weak=bool(base['weak'][q]),core_weight=float(base['cw'][q]),query_position=t,next_official_session=str(sessions[t+1].date()),clock_sessions=[] if partial else dates(list(range(t-294,t+2))),original_price_origin='2008-01-02',u0_indices=[],active_indices=[],numerator_positions=numerator,numerator_dates=dates(numerator),current_positions=current,current_dates=dates(current),lag_positions=lag,lag_dates=dates(lag),required_positions=positions,required_dates=dates(positions),cells=[],zero_current_indices=[],zero_lag_indices=[],branch='initial_partial' if partial else None,current_target=None,lag_target=None)

def _branch(e):
    if e['initial_partial']:return 'initial_partial'
    if e['calm']:return 'calm'
    if not e['u0_indices']:return 'empty_domain'
    if e['zero_current_indices'] or e['zero_lag_indices']:return 'zero_risk'
    return 'available'

def finish_statistics(e):
    # Every required instrument is already scanned and measured before this loop.
    e['zero_current_indices']=[c['column_index'] for c in e['cells'] if c['statistics'] is not None and c['statistics']['current_risk']['sd']==0.]
    e['zero_lag_indices']=[c['column_index'] for c in e['cells'] if c['statistics'] is not None and c['statistics']['lag_risk']['sd']==0.]
    e['branch']=_branch(e)
    if e['branch']=='available':
        for c in e['cells']:
            s=c['statistics']
            if s is not None:
                s['current_ratio']=quotient(s['momentum'],s['current_risk']['sd'])
                s['lag_ratio']=quotient(s['momentum'],s['lag_risk']['sd'])
    return e

def _read_prices(frame,symbol,dates,bound):
    require(symbol in frame.columns,'Required source column')
    out=[]
    for d in dates:
        timestamp=core.iso(d);require(timestamp in frame.index,'Required official source row')
        value=finite(frame.at[timestamp,symbol],'Required source price');require(value>0.,'Positive required source price')
        if bound:require(d>=bound['not_before'],'Required history predates visible issuer bound')
        out.append(dict(date=d,value=value))
    return out

def _check_prices(rows,dates,bound):
    require(type(rows) is list and len(rows)==len(dates),'Complete required107 observations')
    values=[]
    for row,d in zip(rows,dates):
        require(type(row) is dict and set(row)=={'date','value'} and row['date']==d and type(row['value']) is float,'Canonical source observation')
        value=finite(row['value'],'Cached required price');require(value>0.,'Positive cached required price')
        if bound:require(d>=bound['not_before'],'Cached source predates visible issuer bound')
        values.append(value)
    return values

CLOCK_KEYS=('query_index','date','initial_partial','potential_numeric_evaluation','symbols','query_position','next_official_session','numerator_positions','numerator_dates','current_positions','current_dates','lag_positions','lag_dates','required_positions','required_dates','required_count','return_count')
IDENTITY_KEYS=('query_index','column_index','date','symbol','issuer_key','not_before','available_from','bound_visible')

def clock_record(e):
    result={k:copy.deepcopy(e[k]) for k in CLOCK_KEYS if k in e}
    result.update(potential_numeric_evaluation=not e['initial_partial'],required_count=0 if e['initial_partial'] else 107,return_count=0 if e['initial_partial'] else 126)
    return {k:result[k] for k in CLOCK_KEYS}

def _consumed(rows,n,label):
    require(type(rows) is list,label+' list');out=[]
    for row in rows:
        require(isinstance(row,dict) and type(row.get('query_index')) is int and row['query_index']>=0,label+' native query index')
        if row['query_index']<n:out.append(row)
    return out

def source_metadata(review,events):
    require(type(review) is dict and review.get('schema')==SOURCE_SCHEMA,'Source metadata schema')
    for k,v in dict(passed=True,metadata_only=True,source_admitted=False,strict_pit=False,panel_first_session='2008-01-02').items():record_check(review.get(k),v,'Source scope '+k)
    record_check(review.get('listing_bounds'),list(peer.BOUNDS.values()),'Original239 listing policy')
    n=len(events);clocks=_consumed(review['clock_events'],n,'Metadata clocks');identities=_consumed(review['identity_cells'],n,'Metadata identities')
    record_check(clocks,[clock_record(e) for e in events],'Complete consumed official clocks')
    wanted=[]
    for e in events:
        for j,symbol in enumerate(e['symbols']):
            if symbol not in peer.BOUNDS:continue
            b=peer.BOUNDS[symbol]
            wanted.append(dict(query_index=e['query_index'],column_index=j,date=e['date'],symbol=symbol,issuer_key=b['issuer_key'],not_before=b['not_before'],available_from=b['available_from'],bound_visible=b['available_from']<=e['date']))
    record_check(identities,wanted,'Complete issuer identity inventory')

def independent_features(panel,meta,calendar,arrays,review,n=None):
    base,frame,sessions,proof,work=original_evidence(panel,meta,calendar,arrays,n)
    result=[event_clock(q,base,sessions) for q in range(len(proof))];source_metadata(review,result)
    for e,p in zip(result,proof):
        if e['initial_partial']:continue
        q=e['query_index'];e['u0_indices']=p['u0_indices'].copy();e['active_indices']=[] if e['calm'] else e['u0_indices'].copy()
        for j,symbol in enumerate(e['symbols']):
            bound=_bound(symbol,e['date']);rows=[];stats=None
            if j in e['active_indices']:
                rows=_read_prices(frame,symbol,e['required_dates'],bound)
                stats=measure([r['value'] for r in rows])
            e['cells'].append(dict(column_index=j,symbol=symbol,original_finite=j in e['u0_indices'],original_raw=p['original_raw'][j],original_rank=p['original_rank'][j],frozen_core=p['frozen_core'][j],rebuilt_core=p['rebuilt_core'][j],core_abs_error=p['core_abs_error'][j],original_sigma63=p['original_sigma63'][j],history_bound=bound,observations=rows,statistics=stats))
        finish_statistics(e)
    return result

def _cached_clock(e,q,base):
    require(type(e) is dict and set(e)==set(EVENT_KEYS),'Feature event27')
    require(type(e['query_index']) is int and e['query_index']==q,'Native feature query index')
    t=e['query_position'];require(type(t) is int and t>=294,'Native official query position')
    d=str(base['days'][q].date());partial=d==INITIAL
    if partial:
        require(e['clock_sessions']==[] and q==0,'Partial empty clock')
        witness=core.iso(e['next_official_session']);require(witness>base['days'][q],'Partial next-session witness')
        wanted=dict(query_index=q,date=d,symbols=list(base['names'][q]),initial_partial=True,calm=bool(base['calm'][q]),weak=bool(base['weak'][q]),core_weight=float(base['cw'][q]),query_position=t,next_official_session=e['next_official_session'],clock_sessions=[],original_price_origin='2008-01-02',u0_indices=[],active_indices=[],numerator_positions=[],numerator_dates=[],current_positions=[],current_dates=[],lag_positions=[],lag_dates=[],required_positions=[],required_dates=[],cells=[],zero_current_indices=[],zero_lag_indices=[],branch='initial_partial',current_target=None,lag_target=None)
    else:
        cal=e['clock_sessions'];require(type(cal) is list and len(cal)==296,'Complete296 official clock witness')
        ds=[core.iso(x) for x in cal];require(all(d.weekday()<5 for d in ds) and all(a<b for a,b in zip(ds,ds[1:])),'Ordered official clock witness')
        require(ds[294]==base['days'][q],'Clock query date');local=dict(base);local['days']=[base['days'][q]];local['names']=[base['names'][q]];local['calm']=base['calm'][q:q+1];local['weak']=base['weak'][q:q+1];local['cw']=base['cw'][q:q+1]
        wanted=event_clock(0,local,ds);wanted['query_index']=q;wanted['query_position']=t
        for k in ('numerator_positions','current_positions','lag_positions','required_positions'):wanted[k]=[x+t-294 for x in wanted[k]]
    for k in EVENT_KEYS:
        if k not in ('u0_indices','active_indices','cells','zero_current_indices','zero_lag_indices','branch','current_target','lag_target'):record_check(e[k],wanted[k],'Cached clock/state '+k)
    return wanted

def cached_features(arrays,meta,features,n=None,require_raw=False):
    n=_n(meta,n);base=core.original_base(arrays,meta,n);width=base['width'];require(type(features) is list and len(features)>=n,'Complete feature prefix')
    out=[]
    for q in range(n):
        e=features[q];expected=_cached_clock(e,q,base)
        if e['initial_partial']:
            record_check(unranked_prefix([e],1)[0],expected,'Exact partial evidence')
        else:
            cells=e['cells'];require(type(cells) is list and len(cells)==width,'All original source cells');raw=[]
            for j,c in enumerate(cells):
                require(type(c) is dict and set(c)==set(CELL_KEYS),'Cell12')
                require(type(c['column_index']) is int and c['column_index']==j and c['symbol']==base['names'][q][j],'Cell identity')
                require(type(c['original_finite']) is bool,'Original finite flag')
                if c['original_raw'] is None:
                    require(not c['original_finite'],'Missing raw finite flag');raw.append(math.nan)
                else:
                    require(type(c['original_raw']) is float and c['original_finite'],'Canonical original raw');raw.append(finite(c['original_raw'],'Original raw'))
                sigma=c['original_sigma63'];require(sigma is None or type(sigma) is float and math.isfinite(sigma) and sigma>=0.,'Original sigma evidence')
            rr,U=core.independent_rank(raw);expected['u0_indices']=U;expected['active_indices']=[] if expected['calm'] else U.copy()
            record_check(e['u0_indices'],U,'Exact original U0');record_check(e['active_indices'],expected['active_indices'],'Exact preE active domain')
            for j,c in enumerate(cells):
                bound=_bound(c['symbol'],e['date']);rebuilt=float(base['cw'][q]*rr[j]);frozen=float(base['core'][q,j]);error=abs(rebuilt-frozen)
                require(error<=C_SOURCE_ATOL,'Cached original weighted core');sigma=c['original_sigma63'];require(expected['calm'] or j not in U or sigma is not None and sigma>0.,'Noncalm finite original denominator')
                stats=None;rows=[]
                if j in expected['active_indices']:
                    values=_check_prices(c['observations'],expected['required_dates'],bound);rows=copy.deepcopy(c['observations']);stats=measure(values)
                else:record_check(c['observations'],[],'No new source outside active U0');require(c['statistics'] is None,'No statistics outside active U0')
                expected['cells'].append(dict(column_index=j,symbol=c['symbol'],original_finite=j in U,original_raw=None if j not in U else raw[j],original_rank=float(rr[j]),frozen_core=frozen,rebuilt_core=rebuilt,core_abs_error=error,original_sigma63=sigma,history_bound=bound,observations=rows,statistics=stats))
            finish_statistics(expected);record_check(unranked_prefix([e],1)[0],expected,'Cached canonical107 statistics and zero policy')
        if require_raw:require(e['current_target'] is None and e['lag_target'] is None,'Source gate before targets')
        out.append(expected)
    return out

def unranked_prefix(features,n):
    require(type(n) is int and type(features) is list and 0<=n<=len(features),'Consumed raw prefix');out=copy.deepcopy(features[:n])
    for e in out:e['current_target']=e['lag_target']=None
    return out

def finalize(features,width):
    out=copy.deepcopy(features)
    for e in out:
        current=[None]*width;lag=[None]*width
        if e['branch']=='available':
            ids=e['u0_indices'];cells=e['cells']
            for target,key in ((current,'current_ratio'),(lag,'lag_ratio')):
                ranked=ranks([cells[j]['statistics'][key] for j in ids])
                for j,r in zip(ids,ranked):target[j]=r
        e['current_target']=current;e['lag_target']=lag
    return out

def feature_counts(features,arrays,width):
    c=dict.fromkeys(COUNT_FIELDS,0);c.update(events=len(features),symbols_per_event=width,eligible_cells=int(np.asarray(arrays['eligible'])[:len(features)].sum()),original_episode_calls=3)
    for e in features:
        partial=e['initial_partial'];c['initial_partial_events']+=int(partial)
        if partial:continue
        c['calm_events']+=int(e['calm']);c['active_events']+=int(not e['calm']);c['original_core_cells']+=width;c['original_finite_cells']+=len(e['u0_indices']);H=len(e['active_indices'])
        c['required_cells']+=H;c['price_observations']+=107*H;c['current_return_observations']+=63*H;c['lag_return_observations']+=63*H;c['current_risk_statistics']+=H;c['lag_risk_statistics']+=H
        c['zero_current_cells']+=len(e['zero_current_indices']);c['zero_lag_cells']+=len(e['zero_lag_indices']);c['zero_risk_fallback_events']+=int(e['branch']=='zero_risk');c['empty_domain_events']+=int(e['branch']=='empty_domain');c['available_events']+=int(e['branch']=='available')
        if e['branch']=='available':c['ratio_evaluations']+=2*len(e['u0_indices']);c['ranked_cells']+=len(e['u0_indices'])
    return c

def _original_work(features,width):
    return dict(original_source_price_observations=sum((e['query_position']+1)*width for e in features if not e['initial_partial']),original_source_cells=sum(width for e in features if not e['initial_partial']))

def feature_receipt(features,arrays,width,source_rebuilt):
    c=feature_counts(features,arrays,width)
    return dict(**{k:True for k in FEATURE_FLAGS},**work_receipt(len(features),slots=c['price_observations'],required=c['required_cells'],ratios_count=c['ratio_evaluations'],rebuilt=source_rebuilt,original=_original_work(features,width) if source_rebuilt else None),feature_counts=c,full_features_supplied=True,source_rebuilt=source_rebuilt,source_raw_targets_none=True,candidate_selection_computed=False,numerical_tolerance=0.,rank_tolerance=0.)

def verify_features(panel,meta,calendar,arrays,review,features,n=None):
    wanted=independent_features(panel,meta,calendar,arrays,review,n)
    require(type(features) is list and len(features)>=len(wanted),'Full supplied source features');record_check(features[:len(wanted)],wanted,'Independent full source features')
    return feature_receipt(wanted,arrays,np.asarray(arrays['eligible']).shape[1],True)

def verify_features_cached(arrays,meta,features,n=None):
    wanted=cached_features(arrays,meta,features,n,require_raw=True)
    return feature_receipt(wanted,arrays,np.asarray(arrays['eligible']).shape[1],False)

def incumbent_receipt(n):
    return dict(**{k:True for k in INCUMBENT_FLAGS},**work_receipt(n,episodes=1),ordered_choices_checked=n,numerical_tolerance=0.,candidate_selection_computed=False)

def verify_incumbent(arrays,meta,orders,scores,pre_scores,n=None):
    n=_n(meta,n);core.verify_incumbent(arrays,meta,orders,scores,pre_scores,n);return incumbent_receipt(n)

def orders_from_choices(scores,choices,meta):
    return {arm:[dict(date=meta['dates'][q],end=meta['ends'][q],signal_date=meta['dates'][q],symbols=[meta['symbols'][q][j] for j in ids],scores=[float(scores[arm][q,j]) for j in ids]) for q,ids in enumerate(choices[arm])] for arm in ARMS}

def _selection(arrays,meta,features,n=None,validated=False):
    n=_n(meta,n);base=core.original_base(arrays,meta,n);width=base['width']
    raw=unranked_prefix(features,n) if validated else cached_features(arrays,meta,features,n)
    final=finalize(raw,width);record_check(features[:n],final,'Full canonical targets')
    counts=feature_counts(final,arrays,width);scores={};pres={};choices={};deltas={};minimum=None;max_error=0.
    for arm,key in zip(ARMS,(None,'current_target','lag_target')):
        pre=base['P'].copy();delta=np.zeros_like(pre)
        if key:
            for q,e in enumerate(final):
                if e['branch']=='available':
                    for j in e['u0_indices']:
                        delta[q,j]=finite(float(float(base['cw'][q]*e[key][j])-base['core'][q,j]),'Full core delta')
            nonzero=delta!=0.;pre[nonzero]+=delta[nonzero]
        score,picked=core.episode(pre,base);scores[arm]=score;pres[arm]=pre;choices[arm]=picked;deltas[arm]=delta
        for q in range(n):
            eligible=np.flatnonzero(base['E'][q]).tolist();ranked=sorted(eligible,key=lambda j:(-pre[q,j],base['names'][q][j]))
            boundary=set(j for j in ranked[2:5] if pre[q,ranked[2]]-pre[q,j]<=.02) if len(ranked)>=3 else set()
            for j in range(width):
                weights=[float(base['cw'][q]),.025*float(base['calm'][q]),.05*float(base['weak'][q]),.01*float(base['A'][q,j]),float(base['budget'][q]) if j in boundary else 0.]
                remaining=1.-weights[0]
                for w in weights[1:]:remaining-=w
                err=abs(math.fsum(weights+[remaining])-1.);require(remaining>=.38-1e-12 and err<=1e-12,'Original complete budget retained')
                minimum=remaining if minimum is None else min(minimum,remaining);max_error=max(max_error,err)
    events=[]
    for q,e in enumerate(final):
        selected={arm:choices[arm][q] for arm in ARMS}
        counts['current_nonzero_deltas']+=int(np.count_nonzero(deltas[ARMS[1]][q]));counts['lag_nonzero_deltas']+=int(np.count_nonzero(deltas[ARMS[2]][q]))
        counts['current_changed_holdings']+=int(set(selected[ARMS[1]])!=set(selected[ARMS[0]]));counts['lag_changed_holdings']+=int(set(selected[ARMS[2]])!=set(selected[ARMS[0]]));counts['candidate_control_changed_holdings']+=int(set(selected[ARMS[2]])!=set(selected[ARMS[1]]))
        events.append(dict(query_index=q,date=e['date'],initial_partial=e['initial_partial'],calm=e['calm'],weak=e['weak'],core_weight=e['core_weight'],branch=e['branch'],eligible_indices=np.flatnonzero(base['E'][q]).tolist(),u0_indices=e['u0_indices'].copy(),original_support=base['support'][q].tolist(),current_deltas=deltas[ARMS[1]][q].tolist(),lag_deltas=deltas[ARMS[2]][q].tolist(),choices=selected,pre_order=sorted(np.flatnonzero(base['E'][q]).tolist(),key=lambda j:(-base['P'][q,j],base['names'][q][j]))))
    gate=dict(passed=True,synthetic_no_callback=True)
    audit=dict(schema='monthly_risk_clock263_decisions_v1',counts=counts,events=events,incumbent_gate=gate.copy(),source_gate=gate.copy())
    return scores,choices,pres,audit,dict(minimum_remaining_support=minimum,maximum_budget_sum_error=max_error)

def selection_oracle(arrays,meta,features,n=None):return _selection(arrays,meta,features,n)[:4]

def _verify_selection(arrays,meta,features,orders,scores,pre_scores,audit,n,validated=False,source_rebuilt=False):
    es,ch,ep,ea,budget=_selection(arrays,meta,features,n,validated);n=len(ea['events']);width=es[ARMS[0]].shape[1]
    record_check(orders,orders_from_choices(es,ch,meta),'Exact ordered choices')
    for got,want,label in ((scores,es,'Final scores'),(pre_scores,ep,'Core replacement pre-scores')):
        require(type(got) is dict and set(got)==set(ARMS),'Three '+label+' arms')
        for arm in ARMS:exact_array(got[arm],want[arm],label+'/'+arm)
    require(type(audit) is dict and set(audit)==set(ea),'Decision audit5')
    default=dict(passed=True,synthetic_no_callback=True)
    if audit['incumbent_gate']!=default:
        record_check(audit['incumbent_gate'],incumbent_receipt(n),'Full incumbent gate proof');ea['incumbent_gate']=audit['incumbent_gate']
    if audit['source_gate']!=default:
        supplied=audit['source_gate'];require(type(supplied) is dict and type(supplied.get('source_rebuilt')) is bool,'Source gate scope')
        record_check(supplied,feature_receipt(features[:n],arrays,width,supplied['source_rebuilt']),'Full source gate proof');ea['source_gate']=supplied
    record_check(audit,ea,'Full decision audit')
    c=ea['counts']
    return dict(**{k:True for k in SELECTION_FLAGS},**work_receipt(n,slots=c['price_observations'],required=c['required_cells'],ratios_count=c['ratio_evaluations'],domains=2*c['available_events'],scalars=2*c['ranked_cells'],episodes=3,rebuilt=source_rebuilt,original=_original_work(features[:n],width) if source_rebuilt else None),**budget,counts=c,ordered_choices_checked=3*n,full_features_supplied=True,full_scores_supplied=True,full_pre_scores_supplied=True,full_audit_supplied=True,numerical_tolerance=0.,rank_tolerance=0.)

def verify_selection(arrays,meta,features,orders,*,scores,pre_scores,audit,n=None):
    return _verify_selection(arrays,meta,features,orders,scores,pre_scores,audit,n)

def verify_prepared(panel,meta,calendar,arrays,review,features,orders,*,scores,pre_scores,audit,n=None):
    wanted=independent_features(panel,meta,calendar,arrays,review,n)
    record_check(unranked_prefix(features,len(wanted)),wanted,'Prepared full source evidence')
    receipt=_verify_selection(arrays,meta,features,orders,scores,pre_scores,audit,len(wanted),validated=True,source_rebuilt=True)
    receipt.update({k:True for k in PREPARE_FLAGS});return receipt

def audit_outcomes(px,opening,orders,exports,results,meta):
    mapping=dict(zip(ARMS,old.ARMS));mapped=lambda x:{mapping[k]:v for k,v in x.items()}
    require(set(orders)==set(exports)==set(results)==set(ARMS),'Outcome arm identities')
    receipt=old.audit_outcomes(px,opening,mapped(orders),mapped(exports),mapped(results),meta)
    receipt['paths']={a:receipt['paths'][mapping[a]] for a in ARMS};return receipt
