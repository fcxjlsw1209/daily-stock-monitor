"""262 independent source, canonical risk and original-funded selection proofs.

Only frozen independent original-score/core/accounting helpers are imported.
No new producer/host, source loader, actual-data entry point or model fitter.
"""
import copy
import hashlib
import importlib.util
import json
import math
from numbers import Real
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'work/monthly_peer_path252_independent_v2.py'
PRIOR_SHA='02d5d53a4f4867057f58f8cd6295b7838a34b75d6fbc477e6a49016a9a3a282b'
POLICY=ROOT/'work/tracking_risk262_listing_policy.json'
POLICY_SHA='dbb8821baa54ddee961a6d06d30bd478b5ff5b18beae3f899cd92466b4ebae92'
SOURCE_SCHEMA='tracking_risk262_source_contract_v1'
ARMS=('incumbent_local67','sector_total_risk_control','sector_tracking_risk_candidate')
EVENT_KEYS=('query_index','date','symbols','initial_partial','query_position','next_official_session','clock_sessions','numerator_positions','numerator_dates','risk_positions','risk_dates','required_positions','required_dates','stock_indices','mature_indices','supported_indices','cells','total_target','tracking_target')
CELL_KEYS=('column_index','symbol','issuer_key','original_sector','sector','mapping_origin','mapping_repair','mapping_evidence','history_bound','path_history_bound','history_status','own_observations','sector_observations','statistics')
OBS_KEYS=('date','value','status','reason')
STAT_KEYS=('schema','own_momentum','sector_momentum','relative_momentum','own_returns','sector_returns','tracking_returns','total_risk','tracking_risk','total_ratio','tracking_ratio','supported')
RISK_KEYS=('mean','centered','squares','square_sum','variance','sd','constant')
COUNT_FIELDS=('events','symbols_per_event','eligible_cells','initial_partial_events','stock_cells','mature_cells','warmup_cells','unmapped_warmup_cells','supported_cells','zero_total_cells','zero_tracking_cells','own_price_observations','sector_price_observations','stock_return_observations','sector_return_observations','tracking_return_observations','total_risk_statistics','tracking_risk_statistics','total_nonzero_deltas','tracking_nonzero_deltas','total_changed_holdings','tracking_changed_holdings','candidate_control_changed_holdings','original_episode_calls','new_model_fits','predictive_model_fits')
DECISION_EVENT_KEYS=('query_index','date','initial_partial','eligible_indices','supported_indices','original_support','total_deltas','tracking_deltas','choices','calm','weak','core_weight','pre_order')
MAPPING_KEYS=('query_index','column_index','date','symbol','issuer_key','original_sector','sector','mapping_origin','mapping_repair','mapping_evidence')
CLOCK_KEYS=('query_index','date','initial_partial','numeric_evaluation_required','symbols','stock_indices','query_position','next_official_session','numerator_positions','numerator_dates','risk_positions','risk_dates','required_positions','required_dates')
BOUND_KEYS=('symbol','issuer_key','not_before','available_from','kind','evidence_id')
FACTOR_BOUNDS={'VOX':'2004-09-23','XLC':'2018-06-18','XLRE':'2015-10-07'}
INITIAL='2016-08-17'
CONTINUOUS_ATOL=CONTINUOUS_RTOL=2e-10
FEATURE_FLAGS=('passed','independent_source_evidence_checked','full_stock_inventory_preserved','official_65_price_clock_checked','all_stocks_before_eligibility','known_sector_warmup_scanned','whole_issuer_identity_checked','visible_history_bounds_checked','original_maturity_preserved','missing_mature_source_fails','canonical_numerator_checked','canonical_risk_statistics_checked','zero_risk_common_domain_checked','signed_relative_momentum_preserved','no_future_or_outcome_inputs','descriptive_counts_checked','exact_discrete')
INCUMBENT_FLAGS=('passed','full_scores_supplied','full_pre_scores_supplied','independent_original_pre_checked','exact_incumbent_copy_checked','original_eligibility_unchanged','single_original_episode')
SELECTION_FLAGS=('passed','independent_original_pre_checked','original_eligibility_unchanged','original_support_checked','original_core_support_preserved','funded_five_percent_formula_checked','finite_zero_target_is_valid','original_budget_checked','calm_nonweak_extension_budget_checked','shared_rank_domain_checked','canonical_percentile_ties_checked','original_effects_unchanged','one_episode_per_arm','no_final_global_rerank','initial_partial_unchanged','full_audit_checked','exact_scores_and_choices_checked','exact_discrete')
PREPARE_FLAGS=tuple(dict.fromkeys(FEATURE_FLAGS+SELECTION_FLAGS+('independent_source_rebuild',)))
WORK_FIELDS=('full_source_slot_inspections','cached_slot_validations','statistic_evaluations','rank_evaluations','original_episode_calls')


def load_pinned(path,want,name):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=want:raise ValueError('Frozen dependency changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=load_pinned(PRIOR,PRIOR_SHA,'tracking262_original_independent')
peer=old.peer
core=old.old
require,record_check,exact_array=peer.require,peer.record_check,peer.exact_array
ORIGINAL_SOURCE_FLAGS=old.ORIGINAL_SOURCE_FLAGS
INCUMBENT_GATE_FLAGS=INCUMBENT_FLAGS
PATHS,ACCOUNTING_SHA=old.PATHS,old.ACCOUNTING_SHA

def accounting():return old.accounting()
def dependency_files():
    require(hashlib.sha256(POLICY.read_bytes()).hexdigest()==POLICY_SHA,'Early listing evidence policy changed')
    return tuple(sorted(set(old.dependency_files())|{PRIOR,POLICY,Path(__file__).resolve()}))
def listing_policy():
    require(hashlib.sha256(POLICY.read_bytes()).hexdigest()==POLICY_SHA,'Early listing policy digest')
    p=json.loads(POLICY.read_text());require(p['schema']=='tracking_risk262_early_listing_policy_v1' and p['source_admitted'] is False,'Early policy scope')
    return p

def finite(x,label='number'):
    require(isinstance(x,Real) and not isinstance(x,(bool,np.bool_)),'Real nonboolean '+label)
    try:v=float(x)
    except (ValueError,TypeError,OverflowError) as exc:raise ValueError('Binary64 '+label) from exc
    require(math.isfinite(v),'Finite '+label)
    if x!=0 and v==0.:raise ArithmeticError('Nonzero conversion underflow '+label)
    return v

def quotient(a,b):
    v=float(a/b)
    if not math.isfinite(v) or a!=0. and v==0.:raise ArithmeticError('Nonfinite/underflow quotient')
    return v

def risk63(values):
    require(len(values)==63,'Exactly63 risk observations')
    x=[finite(v,'risk value') for v in values]
    constant=all(v==x[0] for v in x)
    if constant:return dict(mean=x[0],centered=[0.]*63,squares=[0.]*63,square_sum=0.,variance=0.,sd=0.,constant=True)
    mean=quotient(math.fsum(x),63.);centered=[finite(v-mean,'centered return') for v in x];sq=[]
    for v in centered:
        z=float(v*v)
        if not math.isfinite(z) or v!=0. and z==0.:raise ArithmeticError('Square overflow/underflow')
        sq.append(z)
    ss=finite(math.fsum(sq),'square sum');var=quotient(ss,62.);sd=finite(math.sqrt(var),'standard deviation')
    require(ss>0. and var>0. and sd>0.,'Nonconstant risk strictly positive')
    # Algebraically distinct real-arithmetic identity is a side check only.
    anchor=x[0];shifted=[v-anchor for v in x];mu=math.fsum(shifted)/63.
    side=math.fsum((v-mu)*(v-mu) for v in shifted)/62.
    require(math.isfinite(side) and abs(var-side)<=CONTINUOUS_ATOL+CONTINUOUS_RTOL*abs(var),'Independent shifted variance identity')
    return dict(mean=mean,centered=centered,squares=sq,square_sum=ss,variance=var,sd=sd,constant=False)

def measure(own65,sector65):
    require(len(own65)==len(sector65)==65,'Exactly65 chronological distinct prices')
    own=[finite(v,'own price') for v in own65];sector=[finite(v,'sector price') for v in sector65]
    require(all(v>0. for v in own+sector),'Positive prices')
    # slot0=t-252; slots1..64=t-63..t; t-42 is slot22.
    gi=quotient(own[22],own[0]);gb=quotient(sector[22],sector[0]);require(gi>0. and gb>0.,'Positive endpoint gross')
    mi=finite(gi-1.);mb=finite(gb-1.);Gi=finite(1.+mi);Gb=finite(1.+mb)
    require(Gi>0. and Gb>0.,'Positive reconstructed gross')
    ratio=quotient(Gi,Gb);require(ratio>0.,'Positive relative gross');relative=finite(ratio-1.)
    def returns(p):
        r=[]
        for a,b in zip(p[2:],p[1:-1]):
            g=quotient(a,b);require(g>0.,'Positive daily gross');r.append(finite(g-1.))
        return r
    ri=returns(own);rb=returns(sector);d=[finite(a-b,'tracking return') for a,b in zip(ri,rb)]
    total=risk63(ri);tracking=risk63(d);support=total['sd']>0. and tracking['sd']>0.
    tr=quotient(relative,total['sd']) if support else None
    rr=quotient(relative,tracking['sd']) if support else None
    return dict(schema='monthly_tracking_risk262_statistics_v1',own_momentum=mi,sector_momentum=mb,relative_momentum=relative,own_returns=ri,sector_returns=rb,tracking_returns=d,total_risk=total,tracking_risk=tracking,total_ratio=tr,tracking_ratio=rr,supported=support)

def ranks(values):
    xs=[finite(v,'rank input') for v in values];N=len(xs)
    return [(float(2*sum(y<x for y in xs)+sum(y==x for y in xs)+1)/2.)/float(N)-.5 for x in xs]

def _calendar(calendar):return calendar['sessions'] if type(calendar) is dict else calendar

def event_clock(q,day,names,stocks,t,sessions):
    partial=day==INITIAL
    require(type(t) is int and t>=0 and t+1<len(sessions),'Query and next-session witness')
    require(sessions[t]==day,'Query clock identity')
    if partial:require(q==0,'Named partial only first')
    else:require(t>=252 and sessions[t+1][:7]!=day[:7],'Official monthend/252 history')
    ep=[] if partial else [t-252,t-42];rp=[] if partial else list(range(t-63,t+1));up=[] if partial else [t-252]+rp
    return dict(query_index=q,date=day,initial_partial=partial,numeric_evaluation_required=not partial,symbols=names.copy(),stock_indices=stocks,query_position=t,next_official_session=sessions[t+1],numerator_positions=ep,numerator_dates=[sessions[k] for k in ep],risk_positions=rp,risk_dates=[sessions[k] for k in rp],required_positions=up,required_dates=[sessions[k] for k in up])

def bounds_for(symbol,issuer,day,partial=False):
    b=peer.BOUNDS.get(symbol)
    if partial or b is None or b['issuer_key']!=issuer or day<b['available_from']:return None,None
    path=b
    if symbol=='SPCX' and day>='2026-06-16':path=listing_policy()['path_bound_override']
    return copy.deepcopy(b),copy.deepcopy(path)

def _rows(rows,n,label):
    require(type(rows) is list,label+' list');result=[]
    for row in rows:
        require(isinstance(row,dict) and type(row.get('query_index')) is int and row['query_index']>=0,label+' query index')
        if row['query_index']<n:result.append(row)
    return result

def source_metadata(review,meta,days,names,etfs,sessions,positions):
    n=len(days)
    peer.subset(review,dict(schema=SOURCE_SCHEMA,metadata_only=True,source_admitted=False,strict_pit=False),'Source contract scope')
    record_check(review['listing_bounds'],list(peer.BOUNDS.values()),'Original listing policy')
    require(type(review.get('listing_policy252_v2')) is dict,'Late bound evidence required')
    record_check(review['listing_policy252_v2'].get('policy'),old.listing_policy(),'Separate late confirmation evidence')
    require(type(review.get('early_listing_policy')) is dict,'Early bound evidence required')
    early=review['early_listing_policy'];require(early.get('sha256')==POLICY_SHA,'Early bound SHA');record_check(early.get('policy'),listing_policy(),'Early bound complete evidence')
    factors=[dict(available_from=None,evidence_clock_certified=False,inception_date=d,symbol=s) for s,d in FACTOR_BOUNDS.items()]
    record_check(review['factor_listing_bounds'],factors,'Factor existence bounds')
    records=_rows(review['mapping_records'],n,'Mapping');clocks=_rows(review['clock_events'],n,'Clock');ids=_rows(review['listing_identity_cells'],n,'Listing identity')
    expected=[(q,j) for q,ss in enumerate(names) for j,s in enumerate(ss) if s not in etfs]
    require([(r.get('query_index'),r.get('column_index')) for r in records]==expected,'Complete direct-stock mapping inventory')
    require([r['query_index'] for r in clocks]==list(range(n)),'Complete clock inventory')
    expected_ids=[];out=[]
    for q,(day,ss,t) in enumerate(zip(days,names,positions)):
        d=str(day.date());stock=[j for j,s in enumerate(ss) if s not in etfs];e=event_clock(q,d,ss,stock,t,sessions)
        record_check(clocks[q],e,'Metadata official clock');group=[r for r in records if r['query_index']==q]
        issuers=[r.get('issuer_key') for r in group]
        require(all(type(k) is str and k and k.strip()==k for k in issuers) and len(set(issuers))==len(issuers),'Whole issuer unique/nonempty')
        for r in group:
            require(set(r)==set(MAPPING_KEYS),'Mapping ten fields');j=r['column_index']
            require(r['date']==d and r['symbol']==ss[j],'Mapping identity')
            peer.mapping_check(r,d)
            b,p=bounds_for(r['symbol'],r['issuer_key'],d,e['initial_partial'])
            # The identity inventory includes only query-visible original bounds.
            if b:
                before=[x for x in e['required_dates'] if x<p['not_before']]
                proof='262_early_actual_confirmation' if p!=b else 'original239_bound'
                expected_ids.append(dict(query_index=q,column_index=j,date=d,symbol=ss[j],issuer_key=r['issuer_key'],original_history_bound=b,path_history_bound=p,path_bound_proof=proof,structural_warmup=bool(e['numerator_dates'] and e['numerator_dates'][0]<b['not_before']),required_dates_before_bound=before,required_dates_at_or_after_bound=[x for x in e['required_dates'] if x>=p['not_before']]))
        out.append((e,group))
    record_check(ids,expected_ids,'Listing identity/clock evidence')
    return out

def observation(frame,symbol,day):
    if symbol not in frame.columns:return dict(date=day,value=None,status='missing',reason='missing_column')
    d=peer.iso(day)
    if d not in frame.index:return dict(date=day,value=None,status='missing',reason='missing_row')
    raw=frame.at[d,symbol]
    if raw is None or raw is pd.NA or isinstance(raw,Real) and not isinstance(raw,(bool,np.bool_)) and math.isnan(float(raw)):
        return dict(date=day,value=None,status='missing',reason='missing_value')
    value=finite(raw,'source price');require(value>0.,'Positive finite source price')
    return dict(date=day,value=value,status='valid',reason=None)

def check_observations(rows,dates,bound,factor=False):
    require(type(rows) is list and len(rows)==len(dates),'Complete65 source observation inventory')
    values=[]
    for row,d in zip(rows,dates):
        require(type(row) is dict and set(row)==set(OBS_KEYS) and row['date']==d,'Source observation identity')
        before=bound is not None and d<bound
        if row['status']=='missing':
            require(row['value'] is None and row['reason'] in ('missing_value','missing_row','missing_column'),'Missing taxonomy')
            require(before and not factor,'Missing post-bound/mature required price')
            values.append(None)
        else:
            require(row['status']=='valid' and row['reason'] is None and type(row['value']) is float,'Valid source evidence')
            value=finite(row['value']);require(value>0.,'Positive source price');require(not before,'Positive price before identity bound')
            values.append(value)
    return values

def cell_from_record(r,e,own,sector):
    b,p=bounds_for(r['symbol'],r['issuer_key'],e['date'],e['initial_partial'])
    partial=e['initial_partial'];warm=not partial and b is not None and e['numerator_dates'][0]<b['not_before']
    state='initial_partial' if partial else 'official_pre_inception' if warm else 'observed_full_window'
    require(partial or r['sector'] is not None or warm,'Mature mapping unavailable')
    stats=None
    if not partial:
        prices=check_observations(own,e['required_dates'],None if p is None else p['not_before'])
        if r['sector'] is not None:
            require(r['sector'] in peer.SECTORS|{'VOX'} and r['sector'] not in ('XLC','XLRE'),'Registered factor identity')
            values=check_observations(sector,e['required_dates'],FACTOR_BOUNDS.get(r['sector']),True)
        else:require(sector==[],'Unknown factor never dereferenced');values=None
        if not warm:
            require(all(v is not None for v in prices),'Original mature path cannot contain missing')
            stats=measure(prices,values)
    else:require(own==sector==[],'Partial no source values')
    return dict(column_index=r['column_index'],symbol=r['symbol'],issuer_key=r['issuer_key'],original_sector=r['original_sector'],sector=r['sector'],mapping_origin=r['mapping_origin'],mapping_repair=r['mapping_repair'],mapping_evidence=copy.deepcopy(r['mapping_evidence']),history_bound=b,path_history_bound=p,history_status=state,own_observations=copy.deepcopy(own),sector_observations=copy.deepcopy(sector),statistics=stats)

def assemble_event(e,cells,sessions):
    result={k:copy.deepcopy(e[k]) for k in EVENT_KEYS if k in e}
    t=e['query_position'];result.update(clock_sessions=[] if e['initial_partial'] else sessions[t-252:t+2],mature_indices=[c['column_index'] for c in cells if c['history_status']=='observed_full_window'],supported_indices=[c['column_index'] for c in cells if c['statistics'] is not None and c['statistics']['supported']],cells=cells,total_target=None,tracking_target=None)
    return {k:result[k] for k in EVENT_KEYS}

def independent_features(panel,meta,calendar,arrays,review,n=None):
    n,width,days,names,etfs=peer.axes(meta,arrays,n)
    frame,ss,pos=peer.source_axes(panel,_calendar(calendar),days);sessions=[str(d.date()) for d in ss]
    groups=source_metadata(review,meta,days,names,etfs,sessions,pos);out=[]
    for e,group in groups:
        cells=[]
        for r in group:
            own=[] if e['initial_partial'] else [observation(frame,r['symbol'],d) for d in e['required_dates']]
            sec=[] if e['initial_partial'] or r['sector'] is None else [observation(frame,r['sector'],d) for d in e['required_dates']]
            cells.append(cell_from_record(r,e,own,sec))
        out.append(assemble_event(e,cells,sessions))
    return out

def cached_features(arrays,meta,features,n=None,require_raw=False):
    n,width,days,names,etfs=peer.axes(meta,arrays,n);require(type(features) is list and len(features)>=n,'Feature prefix')
    out=[]
    for q,(day,ss) in enumerate(zip(days,names)):
        f=features[q];require(type(f) is dict and set(f)==set(EVENT_KEYS),'Feature event19')
        d=str(day.date());partial=d==INITIAL;t=f['query_position'];stock=[j for j,s in enumerate(ss) if s not in etfs]
        require(type(t) is int and t>=0,'Feature query position');nextday=f['next_official_session'];peer.iso(nextday);require(nextday>d,'Next official witness')
        if partial:
            require(q==0 and f['clock_sessions']==[],'Partial clock');calendar=[d,nextday];e=event_clock(0,d,ss,stock,0,calendar);e['query_position']=t
        else:
            cal=f['clock_sessions'];require(type(cal) is list and len(cal)==254,'Complete clock witness')
            days2=[peer.iso(x) for x in cal];require(all(x.weekday()<5 for x in days2) and all(a<b for a,b in zip(days2,days2[1:])),'Ordered official clock witness')
            require(t>=252 and cal[252]==d and cal[253]==nextday,'Clock witness endpoints')
            e=event_clock(q,d,ss,stock,252,cal)
            for key in ('numerator_positions','risk_positions','required_positions'):e[key]=[x+t-252 for x in e[key]]
            e['query_position']=t
        for key in CLOCK_KEYS:
            if key!='numeric_evaluation_required':record_check(f[key],e[key],'Cached clock '+key)
        require(type(f['cells']) is list and [x.get('column_index') for x in f['cells']]==stock,'All direct-stock cached cells')
        issuers=[];cells=[]
        for c in f['cells']:
            require(type(c) is dict and set(c)==set(CELL_KEYS),'Cell14');j=c['column_index']
            r={k:c[k] for k in MAPPING_KEYS if k in c};r.update(query_index=q,date=d)
            require(r['symbol']==ss[j] and type(r['issuer_key']) is str and r['issuer_key'] and r['issuer_key'].strip()==r['issuer_key'],'Cached issuer identity')
            issuers.append(r['issuer_key']);peer.mapping_check(r,d)
            expected=cell_from_record(r,e,c['own_observations'],c['sector_observations']);record_check(c,expected,'Cached source and canonical statistics');cells.append(expected)
        require(len(set(issuers))==len(issuers),'Whole issuer uniqueness')
        base={k:copy.deepcopy(f[k]) for k in EVENT_KEYS};base['cells']=cells;base['total_target']=base['tracking_target']=None
        record_check(base['mature_indices'],[c['column_index'] for c in cells if c['history_status']=='observed_full_window'],'Mature inventory')
        record_check(base['supported_indices'],[c['column_index'] for c in cells if c['statistics'] is not None and c['statistics']['supported']],'Common support')
        if require_raw:require(f['total_target'] is None and f['tracking_target'] is None,'Source gate must precede both ranks')
        out.append(base)
    return out

def finalize(features,width):
    result=copy.deepcopy(features)
    for e in result:
        total=[None]*width;tracking=[None]*width
        if not e['initial_partial']:
            cells={c['column_index']:c for c in e['cells']};ids=e['supported_indices']
            for target,key in ((total,'total_ratio'),(tracking,'tracking_ratio')):
                ranked=ranks([cells[j]['statistics'][key] for j in ids])
                for j,v in zip(ids,ranked):target[j]=v
        e['total_target']=total;e['tracking_target']=tracking
    return result

def unranked_prefix(features,n):
    out=copy.deepcopy(features[:n])
    for e in out:e['total_target']=e['tracking_target']=None
    return out

def feature_counts(features,arrays,width):
    n=len(features);c=dict.fromkeys(COUNT_FIELDS,0);c.update(events=n,symbols_per_event=width,eligible_cells=int(np.asarray(arrays['eligible'])[:n].sum()),original_episode_calls=3)
    for e in features:
        c['initial_partial_events']+=int(e['initial_partial']);c['stock_cells']+=len(e['stock_indices'])
        c['mature_cells']+=len(e['mature_indices']);c['supported_cells']+=len(e['supported_indices'])
        for cell in e['cells']:
            warm=cell['history_status']=='official_pre_inception';c['warmup_cells']+=int(warm);c['unmapped_warmup_cells']+=int(warm and cell['sector'] is None)
            c['own_price_observations']+=len(cell['own_observations']);c['sector_price_observations']+=len(cell['sector_observations'])
            s=cell['statistics']
            if s is not None:
                c['zero_total_cells']+=int(s['total_risk']['sd']==0.);c['zero_tracking_cells']+=int(s['tracking_risk']['sd']==0.)
                for k in ('stock_return_observations','sector_return_observations','tracking_return_observations'):c[k]+=63
                for k in ('total_risk_statistics','tracking_risk_statistics'):c[k]+=1
    return c

def work_receipt(n,slots=0,stats=0,ranks_count=0,episodes=0,rebuilt=False):
    return dict(checked_events=n,producer_modules_imported=False,computes_returns=False,full_source_slot_inspections=slots if rebuilt else 0,cached_slot_validations=0 if rebuilt else slots,statistic_evaluations=stats,rank_evaluations=ranks_count,original_episode_calls=episodes)

def feature_receipt(features,arrays,width,source_rebuilt):
    c=feature_counts(features,arrays,width);slots=c['own_price_observations']+c['sector_price_observations']
    return dict(**{k:True for k in FEATURE_FLAGS},**work_receipt(len(features),slots,2*c['mature_cells'],rebuilt=source_rebuilt),feature_counts=c,full_features_supplied=True,source_rebuilt=source_rebuilt,source_raw_targets_none=True,candidate_selection_computed=False,numerical_tolerance=0.,continuous_atol=CONTINUOUS_ATOL,continuous_rtol=CONTINUOUS_RTOL,rank_tolerance=0.)

def verify_features(panel,meta,calendar,arrays,review,features,n=None):
    wanted=independent_features(panel,meta,calendar,arrays,review,n);require(type(features) is list and len(features)>=len(wanted),'Full feature prefix')
    record_check(features[:len(wanted)],wanted,'Independent source raw features')
    return feature_receipt(wanted,arrays,np.asarray(arrays['eligible']).shape[1],True)

def verify_features_cached(arrays,meta,features,n=None):
    wanted=cached_features(arrays,meta,features,n,require_raw=True)
    return feature_receipt(wanted,arrays,np.asarray(arrays['eligible']).shape[1],False)

def verify_original_source(panel,meta,calendar,arrays,n=None):
    n=len(meta['dates']) if n is None else n;b=core.original_base(arrays,meta,n);frame,sessions=core.source_axes(panel,_calendar(calendar),n)
    checked=valid=0;err=0.
    for i,day in enumerate(b['days']):
        require(day in sessions,'Original official query');t=sessions.index(day);require(t>=294 and t+1<len(sessions),'Original history/witness')
        if meta['dates'][i]==INITIAL:continue
        require(sessions[t+1].to_period('M')!=day.to_period('M'),'Original monthend')
        names=b['names'][i];require(all(s in frame.columns for s in names),'Full original inventory')
        p=frame.loc[:day];require(len(p)>=295 and p.index[-295:].equals(pd.DatetimeIndex(sessions[t-294:t+1])),'Original official295 clock')
        raw,_=core.original_values(frame,day,names,'2008-01-02')
        chosen=np.asarray([v[0 if b['calm'][i] else 1] for v in raw],float);chosen[~np.isfinite(chosen)]=np.nan
        r,U=core.independent_rank(chosen);diff=np.abs(b['cw'][i]*r-b['core'][i]);require(np.isfinite(diff).all() and (diff<=1e-12).all(),'Original core source parity')
        checked+=len(names);valid+=len(U);err=max(err,float(diff.max()))
    return dict(**{k:True for k in ORIGINAL_SOURCE_FLAGS},**work_receipt(n),original_checked_cells=checked,original_finite_cells=valid,maximum_core_source_absolute_error=err,C_source_atol=1e-12,C_source_rtol=0.,continuous_atol=CONTINUOUS_ATOL,continuous_rtol=CONTINUOUS_RTOL,rank_tolerance=0.,episode_calls=0,new_target_ranks_computed=False,candidate_selection_computed=False)

def incumbent_receipt(n):
    return dict(**{k:True for k in INCUMBENT_FLAGS},**work_receipt(n,episodes=1),ordered_choices_checked=n,numerical_tolerance=0.,candidate_selection_computed=False)

def verify_incumbent(arrays,meta,orders,scores,pre_scores,n=None):
    n=len(meta['dates']) if n is None else n
    core.verify_incumbent(arrays,meta,orders,scores,pre_scores,n)
    return incumbent_receipt(n)

def orders_from_choices(scores,choices,meta):
    return {arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],symbols=[meta['symbols'][i][j] for j in ids],scores=[float(scores[arm][i,j]) for j in ids]) for i,ids in enumerate(choices[arm])] for arm in ARMS}

def _selection(arrays,meta,features,n=None,validated=False):
    n=len(meta['dates']) if n is None else n;base=core.original_base(arrays,meta,n);width=base['width']
    raw=unranked_prefix(features,n) if validated else cached_features(arrays,meta,features,n)
    final=finalize(raw,width);record_check(features[:n],final,'Both final canonical ranks')
    counts=feature_counts(final,arrays,width);scores={};pres={};choices={};deltas={};minimum=None;error=0.
    for arm,key in zip(ARMS,(None,'total_target','tracking_target')):
        pre=base['P'].copy();d=np.zeros_like(pre);hand=np.zeros_like(pre)
        if key:
            for q,e in enumerate(final):
                for j in e['supported_indices']:
                    d[q,j]=float(.05*float(e[key][j]-base['support'][q,j]));hand[q,j]=.05
            require(np.isfinite(d).all(),'Finite funded deltas');nz=d!=0.;pre[nz]+=d[nz]
        score,picked=core.episode(pre,base);pres[arm]=pre;scores[arm]=score;choices[arm]=picked;deltas[arm]=d
        for q in range(n):
            ranked=sorted(np.flatnonzero(base['E'][q]).tolist(),key=lambda j:(-pre[q,j],base['names'][q][j]))
            boundary=set(j for j in ranked[2:5] if pre[q,ranked[2]]-pre[q,j]<=.02) if len(ranked)>=3 else set()
            for j in range(width):
                weights=[float(base['cw'][q]),.025*float(base['calm'][q]),.05*float(base['weak'][q]),.01*float(base['A'][q,j]),float(base['budget'][q]) if j in boundary else 0.,float(hand[q,j])]
                remaining=1.-weights[0]
                for weight in weights[1:]:remaining-=weight
                err=abs(math.fsum(weights+[remaining])-1.);require(remaining>=.33-1e-12 and err<=1e-12,'Complete .01-extension funded budget')
                minimum=remaining if minimum is None else min(minimum,remaining);error=max(error,err)
    events=[]
    for q,e in enumerate(final):
        choices_q={a:choices[a][q] for a in ARMS}
        counts['total_nonzero_deltas']+=int(np.count_nonzero(deltas[ARMS[1]][q]));counts['tracking_nonzero_deltas']+=int(np.count_nonzero(deltas[ARMS[2]][q]))
        counts['total_changed_holdings']+=int(set(choices_q[ARMS[1]])!=set(choices_q[ARMS[0]]));counts['tracking_changed_holdings']+=int(set(choices_q[ARMS[2]])!=set(choices_q[ARMS[0]]));counts['candidate_control_changed_holdings']+=int(set(choices_q[ARMS[2]])!=set(choices_q[ARMS[1]]))
        events.append(dict(query_index=q,date=e['date'],initial_partial=e['initial_partial'],eligible_indices=np.flatnonzero(base['E'][q]).tolist(),supported_indices=e['supported_indices'].copy(),original_support=base['support'][q].tolist(),total_deltas=deltas[ARMS[1]][q].tolist(),tracking_deltas=deltas[ARMS[2]][q].tolist(),choices=choices_q,calm=bool(base['calm'][q]),weak=bool(base['weak'][q]),core_weight=float(base['cw'][q]),pre_order=sorted(np.flatnonzero(base['E'][q]).tolist(),key=lambda j:(-base['P'][q,j],base['names'][q][j]))))
    default=dict(passed=True,synthetic_no_callback=True)
    audit=dict(schema='monthly_tracking_risk262_decisions_v1',counts=counts,events=events,incumbent_gate=default.copy(),source_gate=default.copy())
    return scores,choices,pres,audit,dict(minimum_remaining_support=minimum,maximum_budget_sum_error=error)

def selection_oracle(arrays,meta,features,n=None):return _selection(arrays,meta,features,n)[:4]

def _verify_selection(arrays,meta,features,orders,scores,pre_scores,audit,n,validated=False,source_rebuilt=False):
    es,ch,ep,ea,budget=_selection(arrays,meta,features,n,validated=validated);n=len(ea['events']);width=es[ARMS[0]].shape[1]
    record_check(orders,orders_from_choices(es,ch,meta),'Exact ordered choices')
    for got,wanted,label in ((scores,es,'Final score'),(pre_scores,ep,'Funded pre-score')):
        require(type(got) is dict and set(got)==set(ARMS),'Three '+label+' arms')
        for arm in ARMS:exact_array(got[arm],wanted[arm],label+'/'+arm)
    require(type(audit) is dict and set(audit)==set(ea),'Decision audit5')
    default=dict(passed=True,synthetic_no_callback=True)
    if audit['incumbent_gate']!=default:record_check(audit['incumbent_gate'],incumbent_receipt(n),'Original gate proof');ea['incumbent_gate']=audit['incumbent_gate']
    if audit['source_gate']!=default:
        r=audit['source_gate'];require(type(r) is dict and type(r.get('source_rebuilt')) is bool,'Source gate scope')
        record_check(r,feature_receipt(features[:n],arrays,width,r['source_rebuilt']),'Source gate proof');ea['source_gate']=r
    record_check(audit,ea,'Full decision audit')
    c=ea['counts'];slots=c['own_price_observations']+c['sector_price_observations'];formal=n-c['initial_partial_events']
    return dict(**{k:True for k in SELECTION_FLAGS},**work_receipt(n,slots,2*c['mature_cells'],2*formal,3,source_rebuilt),**budget,counts=c,ordered_choices_checked=3*n,full_features_supplied=True,full_scores_supplied=True,full_pre_scores_supplied=True,full_audit_supplied=True,numerical_tolerance=0.,continuous_atol=CONTINUOUS_ATOL,continuous_rtol=CONTINUOUS_RTOL,rank_tolerance=0.)

def verify_selection(arrays,meta,features,orders,*,scores,pre_scores,audit,n=None):
    return _verify_selection(arrays,meta,features,orders,scores,pre_scores,audit,n)

def verify_prepared(panel,meta,calendar,arrays,review,features,orders,*,scores,pre_scores,audit,n=None):
    raw=independent_features(panel,meta,calendar,arrays,review,n);width=np.asarray(arrays['eligible']).shape[1]
    record_check(unranked_prefix(features,len(raw)),raw,'Full source reconstruction')
    r=_verify_selection(arrays,meta,features,orders,scores,pre_scores,audit,len(raw),validated=True,source_rebuilt=True)
    r.update({k:True for k in PREPARE_FLAGS});return r

def audit_outcomes(px,opening,orders,exports,results,meta):
    # Adapt only arm names. The pinned independent helper owns all9 accounts,
    # 45 slices, 18 training groups and exact per-trade/period/aggregate ZERO fees.
    mapping=dict(zip(ARMS,old.ARMS))
    mapped=lambda x:{mapping[k]:v for k,v in x.items()}
    require(set(orders)==set(exports)==set(results)==set(ARMS),'Outcome arm identities')
    r=old.audit_outcomes(px,opening,mapped(orders),mapped(exports),mapped(results),meta)
    r['paths']={a:r['paths'][mapping[a]] for a in ARMS};return r
