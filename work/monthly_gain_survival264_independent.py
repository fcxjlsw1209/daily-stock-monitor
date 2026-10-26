"""Independent264 gain-survival proofs; no264 producer or host imports.

Original pandas/core/episode/accounting helpers are SHA-pinned independent263.
New gain paths and exact source/selection checks are implemented here. A runtime
capability permits reuse of one actual original-source proof per preparation.
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
PRIOR=ROOT/'work/monthly_risk_clock263_independent.py'
PRIOR_SHA='025b64c86edbc56d35149e023ebe536f6fe6e8c4b27cdcac60d59a9a7ce6138f'
def load_pinned(path,digest,name):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Frozen dependency changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=load_pinned(PRIOR,PRIOR_SHA,'gain_survival264_prior_independent')
peer,core=old.peer,old.core
require,record_check,exact_array=old.require,old.record_check,old.exact_array
ARMS=('incumbent_local67','endpoint_gain_age_control','continuous_gain_survival_candidate')
INITIAL=old.INITIAL
PATHS,ACCOUNTING_SHA=old.PATHS,old.ACCOUNTING_SHA
C_SOURCE_ATOL,C_SOURCE_RTOL=1e-12,0.
def accounting():return old.accounting()
def dependency_files():return tuple(sorted(set(old.dependency_files())|{PRIOR,Path(__file__).resolve()}))
def _calendar(x):return x['sessions'] if type(x) is dict else x
def _n(meta,n):return old._n(meta,n)
def _json(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def _digest(x):return hashlib.sha256(_json(x).encode()).hexdigest()

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
def product(a,b):
    z=float(a*b)
    if not math.isfinite(z) or a!=0. and b!=0. and z==0.:raise ArithmeticError('Multiplication overflow/underflow')
    return z
def summation(xs):
    try:z=math.fsum(xs)
    except (OverflowError,ValueError) as exc:raise ArithmeticError('Invalid sum') from exc
    return finite(z,'sum')
def ranks(values):return old.ranks(values)

STAT_KEYS=('schema','positive_anchor_indices','positive_gains','potential_ages','survival_ages','first_breach_indices','endpoint_retained','gain_sum','survival_weighted_terms','endpoint_weighted_terms','survival_weighted_sum','endpoint_weighted_sum','normalizer','survival_quality','endpoint_quality','original_raw','survival_raw','endpoint_raw','nonpositive_original_preserved','no_positive_gains','survival_comparisons')
def measure(prices253,original_raw):
    require(len(prices253)==253,'Exactly253 source prices')
    p=[finite(x,'required price') for x in prices253];require(all(x>0. for x in p),'Positive required prices')
    original=finite(original_raw,'Original raw');indices=[];gains=[];ages=[];durations=[];breaches=[];retained=[];checks=0
    # Canonical simple-return arithmetic, not log-return weights or price-level
    # anchor selection: a rounded zero daily gain is not a positive anchor.
    returns=[finite(quotient(p[k],p[k-1])-1.,'Simple gain') for k in range(1,253)]
    for k,g in enumerate(returns,1):
        if g<=0.:continue
        breach=None;duration=0
        for later in range(k+1,253):
            checks+=1
            if p[later]<p[k]:breach=later;break
            duration+=1
        require(type(duration) is int and 0<=duration<=252-k<=251,'Exact integer survival age bounds')
        indices.append(k);gains.append(g);ages.append(252-k);durations.append(duration);breaches.append(breach);retained.append(bool(p[-1]>=p[k]))
    gain_sum=summation(gains)
    survival_terms=[product(g,float(d)) for g,d in zip(gains,durations)]
    endpoint_terms=[product(g,float(a)) if keep else 0. for g,a,keep in zip(gains,ages,retained)]
    survival_sum=summation(survival_terms);endpoint_sum=summation(endpoint_terms)
    normalizer=product(252.,gain_sum)
    if not gains:qs=qe=1.
    else:
        require(gain_sum>0. and normalizer>0.,'Positive gain denominator')
        qs=quotient(survival_sum,normalizer);qe=quotient(endpoint_sum,normalizer)
        require(0.<=qs<=1. and 0.<=qe<=1.,'Quality bounds')
    sr=original if original<=0. else product(original,qs)
    er=original if original<=0. else product(original,qe)
    return dict(schema='monthly_gain_survival264_statistics_v1',positive_anchor_indices=indices,positive_gains=gains,potential_ages=ages,survival_ages=durations,first_breach_indices=breaches,endpoint_retained=retained,gain_sum=gain_sum,survival_weighted_terms=survival_terms,endpoint_weighted_terms=endpoint_terms,survival_weighted_sum=survival_sum,endpoint_weighted_sum=endpoint_sum,normalizer=normalizer,survival_quality=qs,endpoint_quality=qe,original_raw=original,survival_raw=sr,endpoint_raw=er,nonpositive_original_preserved=original<=0.,no_positive_gains=not gains,survival_comparisons=checks)

SOURCE_SCHEMA='gain_survival264_metadata_source_contract_v1'
EVENT_KEYS=('query_index','date','symbols','initial_partial','calm','weak','core_weight','query_position','next_official_session','clock_sessions','original_price_origin','u0_indices','active_indices','numerator_positions','numerator_dates','current_positions','current_dates','formation_positions','formation_dates','required_positions','required_dates','cells','endpoint_identity','survival_identity','branch','endpoint_target','survival_target')
CELL_KEYS=old.CELL_KEYS
ORIGINAL_EVENT_KEYS=old.ORIGINAL_EVENT_KEYS
COUNT_FIELDS=('events','symbols_per_event','eligible_cells','initial_partial_events','calm_events','active_events','original_core_cells','original_finite_cells','required_cells','price_observations','gain_return_observations','positive_gain_anchors','survival_comparisons','endpoint_comparisons','gain_statistics','quality_ratio_evaluations','no_positive_gain_cells','nonpositive_original_cells','positive_raw_multiplications','empty_domain_events','available_events','endpoint_identity_events','survival_identity_events','endpoint_ranked_cells','survival_ranked_cells','endpoint_nonzero_deltas','survival_nonzero_deltas','endpoint_changed_holdings','survival_changed_holdings','candidate_control_changed_holdings','original_episode_calls','new_model_fits','predictive_model_fits')
DECISION_EVENT_KEYS=('query_index','date','initial_partial','calm','weak','core_weight','branch','endpoint_identity','survival_identity','eligible_indices','u0_indices','original_support','endpoint_deltas','survival_deltas','choices','pre_order')
ORIGINAL_SOURCE_FLAGS=old.ORIGINAL_SOURCE_FLAGS
FEATURE_FLAGS=('passed','independent_source_evidence_checked','original_full_history_domain_checked','all_U0_before_eligibility','original_etf_domain_preserved','official253_price_clock_checked','visible_history_bounds_checked','required_source_hard_failures_checked','positive_simple_gain_anchors_checked','first_strict_breach_checked','exclusive_survival_duration_checked','matched_endpoint_age_checked','no_positive_neutral_checked','nonpositive_original_preserved','whole_vector_identity_checked','no_future_or_outcome_inputs','descriptive_counts_checked','exact_discrete')
INCUMBENT_FLAGS=old.INCUMBENT_FLAGS
INCUMBENT_GATE_FLAGS=INCUMBENT_FLAGS
SELECTION_FLAGS=tuple(k for k in old.SELECTION_FLAGS if k!='whole_query_zero_policy_checked')+('whole_vector_identity_checked','nonpositive_original_preserved')
PREPARE_FLAGS=tuple(dict.fromkeys(FEATURE_FLAGS+SELECTION_FLAGS+('independent_source_rebuild',)))
WORK_FIELDS=('full_source_slot_inspections','cached_slot_validations','momentum_evaluations','return_evaluations','statistic_evaluations','ratio_evaluations','rank_evaluations','scalar_rank_evaluations','original_episode_calls','original_source_price_observations','original_source_cells','survival_comparisons','endpoint_comparisons','positive_raw_multiplications','positive_gain_anchors','original_source_snapshot_price_observations','original_source_snapshot_hashes')
def work_receipt(n,*,counts=None,domains=0,scalars=0,episodes=0,rebuilt=False,original=None):
    result=dict.fromkeys(WORK_FIELDS,0)
    result.update(checked_events=n,producer_modules_imported=False,computes_returns=False,rank_evaluations=domains,scalar_rank_evaluations=scalars,original_episode_calls=episodes)
    if counts:
        result.update(full_source_slot_inspections=counts['price_observations'] if rebuilt else 0,cached_slot_validations=0 if rebuilt else counts['price_observations'],return_evaluations=counts['gain_return_observations'],statistic_evaluations=counts['gain_statistics'],ratio_evaluations=counts['quality_ratio_evaluations'])
        for k in ('survival_comparisons','endpoint_comparisons','positive_raw_multiplications','positive_gain_anchors'):result[k]=counts[k]
    if original:
        for k in ('original_source_price_observations','original_source_cells'):result[k]=original[k]
    return result

_ISSUED_ORIGINALS={}
_PROOF_TOKEN=object()
def snapshot_scope(meta,calendar,n):
    formal=[q for q in range(n) if meta['dates'][q]!=INITIAL]
    if not formal:return (),()
    sessions=_calendar(calendar);last=sessions.index(meta['dates'][formal[-1]])
    require(sessions[0]=='2008-01-02','Snapshot original origin')
    dates=tuple(sessions[:last+1]);names=tuple(sorted({s for q in formal for s in meta['symbols'][q]}))
    return dates,names
def snapshot_digest(dates,names,columns):
    h=hashlib.sha256(_json(dict(schema='gain_survival264_source_snapshot_v1',dates=dates,symbols=names,dtype='<f8',order='symbol_then_session')).encode())
    for values in columns:h.update(values)
    return h.hexdigest()
def source_snapshot(panel,meta,calendar,n):
    dates,names=snapshot_scope(meta,calendar,n)
    if not dates:return dates,names,(),snapshot_digest(dates,names,())
    frame=panel['adj'];index=pd.DatetimeIndex(dates)
    require(frame.index.is_unique and frame.columns.is_unique,'Snapshot unique source axes')
    require(frame.loc[:index[-1]].index.equals(index) and all(s in frame.columns for s in names),'Snapshot consumed official grid and symbols')
    columns=[]
    # Slice rows and columns BEFORE conversion. No later source cell, including
    # an unconsumed security column, participates in this source checksum.
    for symbol in names:
        values=[]
        for value in frame.loc[index,symbol].tolist():
            if value is None or value is pd.NA:values.append(float('nan'));continue
            require(isinstance(value,Real) and not isinstance(value,(bool,np.bool_)),'Snapshot original real nonboolean')
            try:converted=float(value)
            except (TypeError,ValueError,OverflowError) as exc:raise ValueError('Snapshot original conversion') from exc
            values.append(float('nan') if math.isnan(converted) else converted)
        columns.append(np.asarray(values,dtype='<f8').tobytes())
    columns=tuple(columns);return dates,names,columns,snapshot_digest(dates,names,columns)
class _OriginalProof:
    __slots__=('_events','_digests','_count','_certificate_sha256','_bindings','_panel','_adj','_snapshot','_sealed')
    def __init__(self,certificate,token,bindings,panel,snapshot):
        require(token is _PROOF_TOKEN,'Original proof issuer')
        object.__setattr__(self,'_events',tuple(_json(e) for e in certificate['original_events']))
        object.__setattr__(self,'_digests',tuple(_digest(e) for e in certificate['original_events']))
        object.__setattr__(self,'_count',certificate['checked_events'])
        object.__setattr__(self,'_certificate_sha256',_digest(certificate));object.__setattr__(self,'_bindings',bindings)
        object.__setattr__(self,'_panel',panel);object.__setattr__(self,'_adj',panel['adj']);object.__setattr__(self,'_snapshot',snapshot);object.__setattr__(self,'_sealed',True)
    def __setattr__(self,key,value):raise TypeError('Immutable verified original proof')
    def __deepcopy__(self,memo):return self
def verify_original_source(panel,meta,calendar,arrays,n=None):
    result=old.verify_original_source(panel,meta,calendar,arrays,n)
    for k in WORK_FIELDS:result.setdefault(k,0)
    snapshot=source_snapshot(panel,meta,calendar,result['checked_events']);dates,names,_,sha=snapshot
    result.update(original_source_snapshot_sha256=sha,original_source_snapshot_price_observations=len(dates)*len(names),original_source_snapshot_hashes=int(bool(dates)))
    bindings=tuple(_json(dict(date=meta['dates'][q],symbols=meta['symbols'][q])) for q in range(result['checked_events']))
    _ISSUED_ORIGINALS[id(result)]=(result,_digest(result),bindings,panel,snapshot)
    return result
def validated_original_proof(certificate,n=None,width=None):
    issued=_ISSUED_ORIGINALS.get(id(certificate))
    require(issued is not None and issued[0] is certificate and issued[1]==_digest(certificate),'Actually verified original source receipt required')
    count=certificate['checked_events'];require(n is None or type(n) is int and n==count,'Exact original proof prefix')
    if width is not None:
        require(type(width) is int and width>0,'Original proof width')
        require(all(e['initial_partial'] or len(e['original_rank'])==width for e in certificate['original_events']),'Original proof full width')
    return _OriginalProof(certificate,_PROOF_TOKEN,issued[2],issued[3],issued[4])
def validate_source_snapshot(cap,panel,meta,calendar,n):
    require(type(cap) is _OriginalProof and panel is cap._panel and panel['adj'] is cap._adj,'Original proof same source panel identity')
    dates,names,_,sha=source_snapshot(panel,meta,calendar,n)
    old_dates,old_names,old_columns,_=cap._snapshot
    require(dates==old_dates[:len(dates)] and all(s in old_names for s in names),'Original source snapshot prefix coverage')
    consumed=tuple(old_columns[old_names.index(s)][:8*len(dates)] for s in names)
    require(sha==snapshot_digest(dates,names,consumed),'Original source snapshot values unchanged')
    return dict(original_source_snapshot_sha256=sha,original_source_snapshot_price_observations=len(dates)*len(names),original_source_snapshot_hashes=int(bool(dates)))
def original_prefix(original_proof,meta,arrays,n):
    require(type(original_proof) is _OriginalProof and n<=original_proof._count,'Verified original capability coverage')
    base=core.original_base(arrays,meta,n);events=[json.loads(x) for x in original_proof._events[:n]]
    for q,e in enumerate(events):
        require(_json(dict(date=meta['dates'][q],symbols=meta['symbols'][q]))==original_proof._bindings[q],'Consumed original identity binding')
        require(e['query_index']==q and e['date']==meta['dates'][q] and e['calm'] is bool(base['calm'][q]),'Consumed original proof event')
        if not e['initial_partial']:
            record_check(e['frozen_core'],base['core'][q].tolist(),'Consumed original frozen core')
            require(len(e['original_rank'])==base['width'],'Consumed original width')
    return base,events

def _bound(symbol,day,partial=False):return old._bound(symbol,day,partial)
def _read_prices(frame,symbol,dates,bound):return old._read_prices(frame,symbol,dates,bound)
def _check_prices(rows,dates,bound):return old._check_prices(rows,dates,bound)
def event_clock(q,base,sessions):
    old_event=old.event_clock(q,base,sessions);t=old_event['query_position'];partial=old_event['initial_partial']
    formation=[] if partial else list(range(t-294,t-41));dates=[str(sessions[k].date()) for k in formation]
    e={k:v for k,v in old_event.items() if k not in ('lag_positions','lag_dates','zero_current_indices','zero_lag_indices','current_target','lag_target')}
    e.update(formation_positions=formation,formation_dates=dates,required_positions=formation.copy(),required_dates=dates.copy(),endpoint_identity=False,survival_identity=False,endpoint_target=None,survival_target=None)
    return {k:e[k] for k in EVENT_KEYS}
def finish_statistics(e):
    e['branch']='initial_partial' if e['initial_partial'] else 'calm' if e['calm'] else 'empty_domain' if not e['u0_indices'] else 'available'
    for role in ('endpoint','survival'):
        e[role+'_identity']=e['branch']=='available' and all(e['cells'][j]['statistics'][role+'_raw'].hex()==e['cells'][j]['original_raw'].hex() for j in e['u0_indices'])
    return e
CLOCK_KEYS=('query_index','date','initial_partial','potential_numeric_evaluation','symbols','query_position','next_official_session','numerator_positions','numerator_dates','current_positions','current_dates','formation_positions','formation_dates','required_positions','required_dates','required_count','return_count')
IDENTITY_KEYS=old.IDENTITY_KEYS
def clock_record(e):
    r={k:copy.deepcopy(e[k]) for k in CLOCK_KEYS if k in e};r.update(potential_numeric_evaluation=not e['initial_partial'],required_count=0 if e['initial_partial'] else 253,return_count=0 if e['initial_partial'] else 252)
    return {k:r[k] for k in CLOCK_KEYS}
def source_metadata(review,events):
    require(type(review) is dict and review.get('schema')==SOURCE_SCHEMA,'Source metadata schema')
    for k,v in dict(passed=True,metadata_only=True,source_admitted=False,strict_pit=False,panel_first_session='2008-01-02').items():record_check(review.get(k),v,'Source scope '+k)
    record_check(review.get('listing_bounds'),list(peer.BOUNDS.values()),'Original listing policy')
    n=len(events);record_check(old._consumed(review['clock_events'],n,'Metadata clocks'),[clock_record(e) for e in events],'Consumed official clocks')
    wanted=[]
    for e in events:
        for j,s in enumerate(e['symbols']):
            if s in peer.BOUNDS:
                b=peer.BOUNDS[s];wanted.append(dict(query_index=e['query_index'],column_index=j,date=e['date'],symbol=s,issuer_key=b['issuer_key'],not_before=b['not_before'],available_from=b['available_from'],bound_visible=b['available_from']<=e['date']))
    record_check(old._consumed(review['identity_cells'],n,'Metadata identities'),wanted,'Consumed whole-issuer identities')
def independent_features(panel,meta,calendar,arrays,review,n=None,*,original_proof):
    n=_n(meta,n);base,proof=original_prefix(original_proof,meta,arrays,n)
    validate_source_snapshot(original_proof,panel,meta,calendar,n)
    frame,sessions=core.source_axes(panel,_calendar(calendar),n)
    result=[event_clock(q,base,sessions) for q in range(n)];source_metadata(review,result)
    for e,p in zip(result,proof):
        if e['initial_partial']:continue
        e['u0_indices']=p['u0_indices'].copy();e['active_indices']=[] if e['calm'] else e['u0_indices'].copy()
        for j,s in enumerate(e['symbols']):
            bound=_bound(s,e['date']);rows=_read_prices(frame,s,e['required_dates'],bound) if j in e['active_indices'] else []
            e['cells'].append(dict(column_index=j,symbol=s,original_finite=j in e['u0_indices'],**{k:p[k][j] for k in ('original_raw','original_rank','frozen_core','rebuilt_core','core_abs_error','original_sigma63')},history_bound=bound,observations=rows,statistics=None))
    # Complete all required source validation before any new gain arithmetic.
    for e in result:
        for j in e['active_indices']:
            c=e['cells'][j];c['statistics']=measure([x['value'] for x in c['observations']],c['original_raw'])
        finish_statistics(e)
    return result

def _cached_clock(e,q,base):
    require(type(e) is dict and set(e)==set(EVENT_KEYS),'Feature event schema')
    require(type(e['query_index']) is int and e['query_index']==q,'Native feature query')
    t=e['query_position'];require(type(t) is int and t>=294,'Official query position')
    local={k:base[k] for k in base}
    local.update(days=[base['days'][q]],names=[base['names'][q]],calm=base['calm'][q:q+1],weak=base['weak'][q:q+1],cw=base['cw'][q:q+1])
    if str(base['days'][q].date())==INITIAL:
        require(q==0 and e['clock_sessions']==[],'Named partial empty source')
        # No market or future body is inspected to check the named partial.
        wanted={k:copy.deepcopy(e[k]) for k in EVENT_KEYS}
        wanted.update(date=INITIAL,symbols=list(base['names'][q]),initial_partial=True,calm=bool(base['calm'][q]),weak=bool(base['weak'][q]),core_weight=float(base['cw'][q]),original_price_origin='2008-01-02',u0_indices=[],active_indices=[],cells=[],endpoint_identity=False,survival_identity=False,branch='initial_partial',endpoint_target=None,survival_target=None)
        for k in ('numerator_positions','numerator_dates','current_positions','current_dates','formation_positions','formation_dates','required_positions','required_dates'):wanted[k]=[]
        require(core.iso(e['next_official_session'])>base['days'][q],'Partial next-session witness')
    else:
        cal=e['clock_sessions'];require(type(cal) is list and len(cal)==296,'Complete296 official clock witness')
        ds=[core.iso(x) for x in cal];require(all(d.weekday()<5 for d in ds) and all(a<b for a,b in zip(ds,ds[1:])),'Ordered clock witness')
        require(ds[294]==base['days'][q],'Clock query');wanted=event_clock(0,local,ds);wanted.update(query_index=q,query_position=t)
        for k in ('numerator_positions','current_positions','formation_positions','required_positions'):wanted[k]=[x+t-294 for x in wanted[k]]
    dynamic=('u0_indices','active_indices','cells','endpoint_identity','survival_identity','branch','endpoint_target','survival_target')
    for k in EVENT_KEYS:
        if k not in dynamic:record_check(e[k],wanted[k],'Cached clock/state '+k)
    return wanted
def cached_features(arrays,meta,features,n=None,require_raw=False):
    n=_n(meta,n);base=core.original_base(arrays,meta,n);width=base['width'];require(type(features) is list and len(features)>=n,'Complete feature prefix')
    out=[]
    for q in range(n):
        e=features[q];expected=_cached_clock(e,q,base)
        if not e['initial_partial']:
            cells=e['cells'];require(type(cells) is list and len(cells)==width,'Original full-width source cells');raw=[]
            for j,c in enumerate(cells):
                require(type(c) is dict and set(c)==set(CELL_KEYS),'Cell12')
                require(type(c['column_index']) is int and c['column_index']==j and c['symbol']==base['names'][q][j],'Cell identity')
                require(type(c['original_finite']) is bool,'Original finite flag')
                if c['original_raw'] is None:require(not c['original_finite'],'Missing raw flag');raw.append(math.nan)
                else:require(type(c['original_raw']) is float and c['original_finite'],'Original raw type');raw.append(finite(c['original_raw']))
                sv=c['original_sigma63'];require(sv is None or type(sv) is float and math.isfinite(sv) and sv>=0.,'Original sigma')
            rr,U=core.independent_rank(raw);expected['u0_indices']=U;expected['active_indices']=[] if expected['calm'] else U.copy()
            record_check(e['u0_indices'],U,'Original U0');record_check(e['active_indices'],expected['active_indices'],'Original active U0 beforeE')
            for j,c in enumerate(cells):
                bound=_bound(c['symbol'],e['date']);rebuilt=float(base['cw'][q]*rr[j]);frozen=float(base['core'][q,j]);error=abs(rebuilt-frozen)
                require(error<=C_SOURCE_ATOL,'Original C parity');sv=c['original_sigma63'];require(expected['calm'] or j not in U or sv is not None and sv>0.,'Original finite denominator')
                rows=[]
                if j in expected['active_indices']:_check_prices(c['observations'],expected['required_dates'],bound);rows=copy.deepcopy(c['observations'])
                else:record_check(c['observations'],[],'No source outside active U0');require(c['statistics'] is None,'No statistics outside active U0')
                expected['cells'].append(dict(column_index=j,symbol=c['symbol'],original_finite=j in U,original_raw=None if j not in U else raw[j],original_rank=float(rr[j]),frozen_core=frozen,rebuilt_core=rebuilt,core_abs_error=error,original_sigma63=sv,history_bound=bound,observations=rows,statistics=None))
        out.append(expected)
    for e,expected in zip(features[:n],out):
        for j in expected['active_indices']:
            c=expected['cells'][j];c['statistics']=measure([r['value'] for r in c['observations']],c['original_raw'])
        finish_statistics(expected);record_check(unranked_prefix([e],1)[0],expected,'Cached exact gain path')
        if require_raw:require(e['endpoint_target'] is None and e['survival_target'] is None,'Source gate before ranking')
    return out
def unranked_prefix(features,n):
    require(type(n) is int and type(features) is list and 0<=n<=len(features),'Consumed raw prefix');out=copy.deepcopy(features[:n])
    for e in out:e['endpoint_target']=e['survival_target']=None
    return out
def finalize(features,width):
    out=copy.deepcopy(features)
    for e in out:
        for role in ('endpoint','survival'):
            target=[None]*width
            if e['branch']=='available':
                ids=e['u0_indices'];ranked=ranks([e['cells'][j]['statistics'][role+'_raw'] for j in ids])
                for j,r in zip(ids,ranked):target[j]=r
            e[role+'_target']=target
    return out
def feature_counts(features,arrays,width):
    c=dict.fromkeys(COUNT_FIELDS,0);c.update(events=len(features),symbols_per_event=width,eligible_cells=int(np.asarray(arrays['eligible'])[:len(features)].sum()),original_episode_calls=3)
    for e in features:
        c['initial_partial_events']+=int(e['initial_partial'])
        if e['initial_partial']:continue
        c['calm_events']+=int(e['calm']);c['active_events']+=int(not e['calm']);c['original_core_cells']+=width;c['original_finite_cells']+=len(e['u0_indices']);H=len(e['active_indices'])
        c['required_cells']+=H;c['price_observations']+=253*H;c['gain_return_observations']+=252*H;c['gain_statistics']+=H
        c['empty_domain_events']+=int(e['branch']=='empty_domain');c['available_events']+=int(e['branch']=='available')
        for role in ('endpoint','survival'):
            c[role+'_identity_events']+=int(e[role+'_identity']);c[role+'_ranked_cells']+=H if e['branch']=='available' else 0
        for j in e['active_indices']:
            s=e['cells'][j]['statistics'];A=len(s['positive_anchor_indices']);c['positive_gain_anchors']+=A;c['endpoint_comparisons']+=A;c['survival_comparisons']+=s['survival_comparisons'];c['quality_ratio_evaluations']+=2*int(A>0);c['no_positive_gain_cells']+=int(s['no_positive_gains']);c['nonpositive_original_cells']+=int(s['nonpositive_original_preserved']);c['positive_raw_multiplications']+=2*int(s['original_raw']>0.)
    return c
def proof_fields(original_proof,n):
    require(type(original_proof) is _OriginalProof and n<=original_proof._count,'Full gate verified capability required')
    records=[json.loads(x) for x in original_proof._bindings[:n]];formal=[r for r in records if r['date']!=INITIAL]
    all_dates,all_names,all_columns,_=original_proof._snapshot
    dates=all_dates[:all_dates.index(formal[-1]['date'])+1] if formal else ()
    names=tuple(sorted({s for r in formal for s in r['symbols']}))
    values=tuple(all_columns[all_names.index(s)][:8*len(dates)] for s in names)
    return dict(original_source_proof_sha256=original_proof._certificate_sha256,original_source_proof_events=n,original_source_snapshot_sha256=snapshot_digest(dates,names,values),original_source_snapshot_price_observations=len(dates)*len(names),original_source_snapshot_hashes=int(bool(dates)))
def feature_receipt(features,arrays,width,source_rebuilt,original_proof=None):
    c=feature_counts(features,arrays,width)
    fields=proof_fields(original_proof,len(features)) if source_rebuilt else {}
    result=dict(**{k:True for k in FEATURE_FLAGS},**work_receipt(len(features),counts=c,rebuilt=source_rebuilt),feature_counts=c,full_features_supplied=True,source_rebuilt=source_rebuilt,source_raw_targets_none=True,candidate_selection_computed=False,numerical_tolerance=0.,rank_tolerance=0.)
    result.update(fields);return result
def verify_features(panel,meta,calendar,arrays,review,features,n=None,*,original_proof):
    wanted=independent_features(panel,meta,calendar,arrays,review,n,original_proof=original_proof)
    require(type(features) is list and len(features)>=len(wanted),'Full supplied source');record_check(features[:len(wanted)],wanted,'Independent source gain features')
    return feature_receipt(wanted,arrays,np.asarray(arrays['eligible']).shape[1],True,original_proof)
def verify_features_cached(arrays,meta,features,n=None):
    wanted=cached_features(arrays,meta,features,n,require_raw=True)
    return feature_receipt(wanted,arrays,np.asarray(arrays['eligible']).shape[1],False)
def incumbent_receipt(n):return dict(**{k:True for k in INCUMBENT_FLAGS},**work_receipt(n,episodes=1),ordered_choices_checked=n,numerical_tolerance=0.,candidate_selection_computed=False)
def verify_incumbent(arrays,meta,orders,scores,pre_scores,n=None):
    n=_n(meta,n);core.verify_incumbent(arrays,meta,orders,scores,pre_scores,n);return incumbent_receipt(n)
def orders_from_choices(scores,choices,meta):
    return {arm:[dict(date=meta['dates'][q],end=meta['ends'][q],signal_date=meta['dates'][q],symbols=[meta['symbols'][q][j] for j in ids],scores=[float(scores[arm][q,j]) for j in ids]) for q,ids in enumerate(choices[arm])] for arm in ARMS}
def _selection(arrays,meta,features,n=None,validated=False):
    n=_n(meta,n);base=core.original_base(arrays,meta,n);width=base['width']
    raw=unranked_prefix(features,n) if validated else cached_features(arrays,meta,features,n)
    final=finalize(raw,width);record_check(features[:n],final,'Exact full targets')
    counts=feature_counts(final,arrays,width);scores={};pres={};choices={};deltas={};minimum=None;max_error=0.
    for arm,role in zip(ARMS,(None,'endpoint','survival')):
        pre=base['P'].copy();delta=np.zeros_like(pre)
        if role:
            for q,e in enumerate(final):
                if e['branch']=='available' and not e[role+'_identity']:
                    for j in e['u0_indices']:delta[q,j]=finite(float(float(base['cw'][q]*e[role+'_target'][j])-base['core'][q,j]),'Full core delta')
            nonzero=delta!=0.;pre[nonzero]+=delta[nonzero]
        score,picked=core.episode(pre,base);scores[arm]=score;pres[arm]=pre;choices[arm]=picked;deltas[arm]=delta
        for q in range(n):
            eligible=np.flatnonzero(base['E'][q]).tolist();ordered=sorted(eligible,key=lambda j:(-pre[q,j],base['names'][q][j]));boundary=set(j for j in ordered[2:5] if pre[q,ordered[2]]-pre[q,j]<=.02) if len(ordered)>=3 else set()
            for j in range(width):
                weights=[float(base['cw'][q]),.025*float(base['calm'][q]),.05*float(base['weak'][q]),.01*float(base['A'][q,j]),float(base['budget'][q]) if j in boundary else 0.];remaining=1.-weights[0]
                for w in weights[1:]:remaining-=w
                err=abs(math.fsum(weights+[remaining])-1.);require(remaining>=.38-1e-12 and err<=1e-12,'Original complete budget')
                minimum=remaining if minimum is None else min(minimum,remaining);max_error=max(max_error,err)
    events=[]
    for q,e in enumerate(final):
        selected={a:choices[a][q] for a in ARMS}
        for arm,role in zip(ARMS[1:],('endpoint','survival')):
            counts[role+'_nonzero_deltas']+=int(np.count_nonzero(deltas[arm][q]));counts[role+'_changed_holdings']+=int(set(selected[arm])!=set(selected[ARMS[0]]))
        counts['candidate_control_changed_holdings']+=int(set(selected[ARMS[2]])!=set(selected[ARMS[1]]))
        events.append(dict(query_index=q,date=e['date'],initial_partial=e['initial_partial'],calm=e['calm'],weak=e['weak'],core_weight=e['core_weight'],branch=e['branch'],endpoint_identity=e['endpoint_identity'],survival_identity=e['survival_identity'],eligible_indices=np.flatnonzero(base['E'][q]).tolist(),u0_indices=e['u0_indices'].copy(),original_support=base['support'][q].tolist(),endpoint_deltas=deltas[ARMS[1]][q].tolist(),survival_deltas=deltas[ARMS[2]][q].tolist(),choices=selected,pre_order=sorted(np.flatnonzero(base['E'][q]).tolist(),key=lambda j:(-base['P'][q,j],base['names'][q][j]))))
    gate=dict(passed=True,synthetic_no_callback=True)
    audit=dict(schema='monthly_gain_survival264_decisions_v1',counts=counts,events=events,incumbent_gate=gate.copy(),source_gate=gate.copy())
    return scores,choices,pres,audit,dict(minimum_remaining_support=minimum,maximum_budget_sum_error=max_error)
def selection_oracle(arrays,meta,features,n=None):return _selection(arrays,meta,features,n)[:4]
def _verify_selection(arrays,meta,features,orders,scores,pre_scores,audit,n,validated=False,source_rebuilt=False,original_proof=None):
    es,ch,ep,ea,budget=_selection(arrays,meta,features,n,validated);n=len(ea['events']);width=es[ARMS[0]].shape[1]
    record_check(orders,orders_from_choices(es,ch,meta),'Exact ordered choices')
    for got,want,label in ((scores,es,'Final scores'),(pre_scores,ep,'Pre scores')):
        require(type(got) is dict and set(got)==set(ARMS),'Three arms '+label)
        for arm in ARMS:exact_array(got[arm],want[arm],label+'/'+arm)
    require(type(audit) is dict and set(audit)==set(ea),'Decision audit5');default=dict(passed=True,synthetic_no_callback=True)
    if audit['incumbent_gate']!=default:record_check(audit['incumbent_gate'],incumbent_receipt(n),'Independent incumbent gate');ea['incumbent_gate']=audit['incumbent_gate']
    if audit['source_gate']!=default:
        supplied=audit['source_gate'];require(type(supplied) is dict and type(supplied.get('source_rebuilt')) is bool,'Source receipt scope')
        record_check(supplied,feature_receipt(features[:n],arrays,width,supplied['source_rebuilt'],original_proof),'Full source gate');ea['source_gate']=supplied
    record_check(audit,ea,'Full decision audit');c=ea['counts']
    fields=proof_fields(original_proof,n) if source_rebuilt else {}
    result=dict(**{k:True for k in SELECTION_FLAGS},**work_receipt(n,counts=c,domains=2*c['available_events'],scalars=c['endpoint_ranked_cells']+c['survival_ranked_cells'],episodes=3,rebuilt=source_rebuilt),**budget,counts=c,ordered_choices_checked=3*n,full_features_supplied=True,full_scores_supplied=True,full_pre_scores_supplied=True,full_audit_supplied=True,numerical_tolerance=0.,rank_tolerance=0.)
    result.update(fields);return result
def verify_selection(arrays,meta,features,orders,*,scores,pre_scores,audit,n=None):return _verify_selection(arrays,meta,features,orders,scores,pre_scores,audit,n)
def verify_prepared(panel,meta,calendar,arrays,review,features,orders,*,scores,pre_scores,audit,n=None,original_proof):
    wanted=independent_features(panel,meta,calendar,arrays,review,n,original_proof=original_proof)
    record_check(unranked_prefix(features,len(wanted)),wanted,'Full prepared source evidence')
    result=_verify_selection(arrays,meta,features,orders,scores,pre_scores,audit,len(wanted),validated=True,source_rebuilt=True,original_proof=original_proof)
    result.update({k:True for k in PREPARE_FLAGS});return result
def audit_outcomes(px,opening,orders,exports,results,meta):
    mapping=dict(zip(ARMS,old.ARMS));mapped=lambda x:{mapping[k]:v for k,v in x.items()}
    require(set(orders)==set(exports)==set(results)==set(ARMS),'Outcome arm identities')
    result=old.audit_outcomes(px,opening,mapped(orders),mapped(exports),mapped(results),meta)
    result['paths']={a:result['paths'][mapping[a]] for a in ARMS};return result
