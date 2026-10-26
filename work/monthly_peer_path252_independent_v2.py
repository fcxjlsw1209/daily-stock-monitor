"""Independent 252 v2 source, exact peer-reference, and original-score checks.

No new producer, host, source loader, strategy engine, or actual-data entry point.
Frozen independent metadata, original-score and accounting primitives are pinned.
"""
import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PEER=ROOT/'work/monthly_peer239_independent.py'
PEER_SHA='412c3f1e5f5c185d4eedea5cc954fcbc81737ae370f24d09cbdb96c166340064'
CORE=ROOT/'work/monthly_calendar_clock248_v2_independent.py'
CORE_SHA='c7d172a9abec73f81d1d07ce8a422cd709a4a9bfebd55e6211b471c8d315663c'
ACCOUNT_BASE=ROOT/'work/monthly_reference_gain250_independent.py'
ACCOUNT_BASE_SHA='d61b3243af183da3e65fd1cbe4052e35d8933fb5c2472fc236e29e92832729a4'
PEER_TEST=ROOT/'work/test_monthly_peer239_independent.py'
PEER_TEST_SHA='938d0de8b29e739502107174b44707333830b7e1c49e6df9734a146f2132ff53'
CORE_TEST=ROOT/'work/test_monthly_reference_gain250_independent.py'
CORE_TEST_SHA='8b15f23ba83f36f2d063a87a54a9b76093c703df2c9a1f1f2a7846a4afc4e106'
LISTING_POLICY=ROOT/'work/monthly_peer_path252_v2_LISTING_POLICY.json'
LISTING_POLICY_SHA='919a698dafe5fafc7b0516ac7fb613d1ff5292e0a831f049dc134b8864ca1e91'
V1=ROOT/'work/monthly_peer_path252_independent.py'
V1_SHA='28c638f8092de8a729c9dc2469453eab029c514c4c741b8dd1f3bf996089fb3c'
ARMS=('incumbent_local67','daily_peer_path_relative','endpoint_peer_relative_control')
VALUE_KEYS=('mature','supported','path_target','endpoint_target')
COUNT_FIELDS=('events','stock_cells','mature_cells','supported_cells','price_observations','new_model_fits')
SOURCE_SCHEMA='monthly_peer_path252_daily_source_report_v2'
SOURCE_TOP_KEYS=('schema','status','admitted','events','records','gaps','counts','source_only',
 'original_inventory_preserved','strict_pit','momentum_computed','rankings_computed','orders_computed',
 'returns_computed','model_fits','original_failed_audit_preserved','daily_paths_checked')
SOURCE_EVENT_KEYS=('query_index','date','initial_partial','stock_indices','source_positions','source_dates',
 'mature_indices','supported_indices','path_dates')
LEGACY_SOURCE_RECORD_KEYS=('query_index','column_index','date','symbol','issuer_key','original_sector','sector',
 'mapping_origin','mapping_repair','mapping_evidence','peer_indices','peer_issuer_count','structural_peer_support',
 'initial_partial','source_positions','source_dates','endpoints','gap_reasons','original_gap_reasons',
 'history_status','history_bound','deferred_gap_reasons')
SOURCE_RECORD_KEYS=LEGACY_SOURCE_RECORD_KEYS+('path_observations','path_history_bound')
EVENT_KEYS=('query_index','date','source_dates','source_positions','stock_indices','mature_indices','supported_indices','cells')
CELL_KEYS=('column_index','symbol','issuer_key','sector','history_status','peer_indices','peer_issuer_count',
 'focal_gross','daily_peer_gross','endpoint_peer_gross','path_raw','endpoint_raw','path_target','endpoint_target')
SOURCE_FLAGS=('passed','independent_daily_source_rebuild','full_stock_inventory_preserved',
 'official_211_price_clock_checked','all_mature_sources_before_eligibility','warmup_full_window_checked',
 'visible_history_bounds_checked','whole_issuer_exclusion_checked','complete_peer_set_checked',
 'minimum_two_distinct_peers_checked','positive_nonboolean_prices_checked','mature_missing_hard_failure',
 'source_missing_not_filled','source_counts_checked','no_future_source_values',
 'path_listing_policy_pinned','path_bound_publication_clock_checked','legacy_maturity_and_cohort_preserved',
 'all_mature_daily_prices_required')
FEATURE_FLAGS=('passed','independent_exact_rational_rebuild','daily_source_admission_checked',
 'whole_issuer_exclusion_checked','complete_peer_set_checked','shared_supported_domain_checked',
 'all_stocks_before_eligibility','exact_daily_median_product_checked','exact_endpoint_median_checked',
 'relative_ratio_control_checked','exact_raw_rational_evidence_checked','canonical_percentile_rank_checked',
 'finite_zero_target_is_valid','etfs_partial_warmup_unavailable','empty_domain_noop_checked',
 'full_feature_audit_checked','descriptive_counts_checked','no_future_or_outcome_inputs','path_bound_evidence_checked')
ORIGINAL_SOURCE_FLAGS=('passed','original_core_checked','original_full_history_pandas_checked',
 'original_official_row_clock_checked','original_raw_rank_parity_checked','original_weighted_core_tolerance_checked',
 'original_eligibility_unchanged','named_initial_partial_unchanged','exact_discrete')
SELECTION_FLAGS=('passed','independent_original_pre_checked','original_model_rank_scope_preserved',
 'original_eligibility_unchanged','original_support_checked','funded_five_percent_formula_checked',
 'finite_zero_target_is_valid','calm_nonweak_extension_budget_checked','original_budget_checked',
 'core_weights_unchanged','original_effects_unchanged','one_episode_per_arm','no_second_episode',
 'no_final_global_rerank','initial_partial_unchanged','empty_domain_noop_checked','exact_scores_and_choices_checked')
C_SOURCE_ATOL=1e-12
C_SOURCE_RTOL=0.
CONTINUOUS_ATOL=CONTINUOUS_RTOL=2e-10
PANEL_FIRST_SESSION='2008-01-02'


def load_pinned(path,want,name):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=want:raise ValueError('Frozen independent dependency changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


peer=load_pinned(PEER,PEER_SHA,'path252_peer_metadata')
old=load_pinned(CORE,CORE_SHA,'path252_original_source')
require,record_check,exact_array=peer.require,peer.record_check,peer.exact_array
INCUMBENT_GATE_FLAGS=old.INCUMBENT_GATE_FLAGS
PATHS,ACCOUNTING_SHA=old.PATHS,old.ACCOUNTING_SHA


def accounting():return old.accounting()
def frozen_source_oracle():return old.frozen_source_oracle()
def listing_policy():
    raw=LISTING_POLICY.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==LISTING_POLICY_SHA,'Frozen v2 listing policy changed')
    policy=json.loads(raw)
    require(policy['schema']=='monthly_peer_path252_v2_listing_policy_v1' and policy['source_only'] is True,
            'V2 listing policy scope')
    record_check(policy['original_history_bound'],peer.BOUNDS['SPCX'],'Preserved original SPCX lower bound')
    record_check(policy['path_bound_override'],dict(symbol='SPCX',
      issuer_key='https://companiesmarketcap.com/spacex/marketcap/',not_before='2026-06-12',
      available_from='2026-08-05',kind='issuer_actual_first_trade',evidence_id='spcx_actual_trading_confirmation'),
      'Fixed actual-trading path bound')
    require(policy['publication_date']=='2026-08-04' and policy['strict_pit'] is False,'Conservative disclosure clock')
    return policy


def _path_history_bound(row,policy):
    override=policy['path_bound_override']
    if not row['initial_partial'] and row['symbol']==override['symbol'] and row['issuer_key']==override['issuer_key'] and row['date']>=override['available_from']:
        record_check(row['history_bound'],policy['original_history_bound'],'Override preserves original bound')
        return copy.deepcopy(override)
    return copy.deepcopy(row['history_bound'])


def dependency_files():
    require(hashlib.sha256(V1.read_bytes()).hexdigest()==V1_SHA,'Frozen v1 comparison source changed')
    require(hashlib.sha256(CORE_TEST.read_bytes()).hexdigest()==CORE_TEST_SHA,'Frozen original fixture changed')
    require(hashlib.sha256(PEER_TEST.read_bytes()).hexdigest()==PEER_TEST_SHA,'Frozen peer fixture changed')
    account=load_pinned(ACCOUNT_BASE,ACCOUNT_BASE_SHA,'path252_account_dependencies')
    policy=listing_policy()
    return tuple(sorted(set(peer.dependency_files())|set(old.dependency_files())|set(account.dependency_files())|
        {ROOT/p for p in policy['input_sha256']}|
        {PEER,CORE,ACCOUNT_BASE,PEER_TEST,CORE_TEST,LISTING_POLICY,V1,Path(__file__).resolve()}))


def _shape(panel,meta,calendar,arrays,limit):
    n,width,days,names,etfs=peer.axes(meta,arrays,limit)
    frame,sessions,positions=peer.source_axes(panel,calendar,days)
    return n,width,days,names,etfs,frame,sessions,positions


def original_base(original,meta,n):return old.original_base(original,meta,n)


def verify_original_source(panel,meta,calendar,arrays,n=None):
    n=len(meta['dates']) if n is None else n;base=original_base(arrays,meta,n)
    require(base['width']==49,'Original source gate requires full49 inventory')
    frame,sessions=old.source_axes(panel,calendar,n);checked=finite_count=0;maxerr=0.
    for i,day in enumerate(base['days']):
        require(day in sessions,'Original query on official calendar');t=sessions.index(day)
        require(t>=294 and t+1<len(sessions),'Original history and next-session witness')
        if meta['dates'][i]=='2016-08-17':continue
        require(sessions[t+1].to_period('M')!=day.to_period('M'),'Official month end')
        names=base['names'][i];require(all(s in frame.columns for s in names),'All original49 price columns')
        prefix=frame.loc[:day]
        require(len(prefix)>=295 and prefix.index[-295:].equals(pd.DatetimeIndex(sessions[t-294:t+1])),
                'Original last295 official row clock')
        raw,_=old.original_values(frame,day,names,PANEL_FIRST_SESSION)
        chosen=np.asarray([v[0 if base['calm'][i] else 1] for v in raw],float)
        chosen[~np.isfinite(chosen)]=np.nan;ranks,U=old.independent_rank(chosen)
        rebuilt=base['cw'][i]*ranks;difference=np.abs(rebuilt-base['core'][i])
        require(np.isfinite(difference).all() and (difference<=C_SOURCE_ATOL).all(),'Frozen original weighted core parity')
        checked+=len(names);finite_count+=len(U);maxerr=max(maxerr,float(difference.max()))
    return dict(**{k:True for k in ORIGINAL_SOURCE_FLAGS},checked_events=n,original_checked_cells=checked,
      original_finite_cells=finite_count,maximum_core_source_absolute_error=maxerr,
      C_source_atol=C_SOURCE_ATOL,C_source_rtol=C_SOURCE_RTOL,continuous_atol=CONTINUOUS_ATOL,
      continuous_rtol=CONTINUOUS_RTOL,rank_tolerance=0.,episode_calls=0,new_target_ranks_computed=False,
      candidate_selection_computed=False,producer_modules_imported=False,computes_returns=False)


def verify_incumbent(original,meta,orders,scores,pre_scores,n=None):
    return old.verify_incumbent(original,meta,orders,scores,pre_scores,n)


def _observation(frame,symbol,day):
    """Classify absence separately; every present malformed value is a hard error."""
    if symbol not in frame.columns:return None,'missing_column'
    if day not in frame.index:return None,'missing_row'
    value=frame.at[day,symbol]
    if value is None or value is pd.NA:return None,'missing_value'
    require(isinstance(value,Real) and not isinstance(value,(bool,np.bool_)),'Required real nonboolean price')
    try:f=float(value)
    except (ValueError,TypeError,OverflowError) as exc:raise ValueError('Invalid required price conversion') from exc
    if math.isnan(f):return None,'missing_value'
    require(math.isfinite(f) and f>0.,'Required finite positive price')
    return f,None


def _inspect_path(frame,row,path_dates):
    observations=[];prices=[];errors=[];bound=row['path_history_bound'];status=row['history_status']
    for day in path_dates:
        ds=str(day.date());prebound=bound is not None and ds<bound['not_before']
        try:value,reason=_observation(frame,row['symbol'],day)
        except ValueError as exc:errors.append((ds,str(exc)));continue
        if reason is None:
            if prebound:errors.append((ds,'positive_pre_bound_identity_contradiction'))
            observations.append(dict(date=ds,status='valid',reason=None));prices.append(value)
        else:
            if not (status=='official_pre_inception' and prebound):errors.append((ds,reason))
            observations.append(dict(date=ds,status='missing',reason=reason));prices.append(None)
    require(not errors,'Daily path admission '+row['symbol']+': '+repr(errors))
    return observations,prices


def _source(panel,meta,calendar,arrays,source239,limit=None):
    n,width,days,names,etfs,frame,sessions,positions=_shape(panel,meta,calendar,arrays,limit)
    groups=peer.report_rows(source239,days,names,etfs,sessions,positions)
    policy=listing_policy()
    events=[];records=[];paths={};errors=[];counts=dict.fromkeys(COUNT_FIELDS,0);counts['events']=n
    for i,(d,ss,t,group) in enumerate(zip(days,names,positions,groups)):
        partial=str(d.date())=='2016-08-17';pdays=[] if partial else sessions[t-252:t-41]
        event={k:copy.deepcopy(source239['events'][i][k]) for k in SOURCE_EVENT_KEYS[:-1]}
        event['path_dates']=[str(x.date()) for x in pdays];events.append(event)
        counts['stock_cells']+=len(group)
        for r in group:
            row={k:copy.deepcopy(r[k]) for k in LEGACY_SOURCE_RECORD_KEYS}
            path_bound=_path_history_bound(r,policy)
            row['path_history_bound']=path_bound
            observations=[];prices=[]
            if not partial:
                counts['price_observations']+=211
                try:observations,prices=_inspect_path(frame,row,pdays)
                except ValueError as exc:errors.append(str(exc))
            row['path_observations']=observations
            row={k:row[k] for k in SOURCE_RECORD_KEYS};records.append(row)
            if r['history_status']=='observed_full_window':
                counts['mature_cells']+=1;paths[i,r['column_index']]=prices
            if r['structural_peer_support']:counts['supported_cells']+=1
    require(not errors,'All direct-stock daily source checks: '+'; '.join(errors))
    report=dict(schema=SOURCE_SCHEMA,status='ADMITTED',admitted=True,events=events,records=records,gaps=[],counts=counts,
      source_only=True,original_inventory_preserved=True,strict_pit=False,momentum_computed=False,rankings_computed=False,
      orders_computed=False,returns_computed=False,model_fits=0,original_failed_audit_preserved=True,daily_paths_checked=True)
    return report,paths,(n,width,days,names,etfs,groups)


def independent_source(panel,meta,calendar,arrays,source239,limit=None):
    return _source(panel,meta,calendar,arrays,source239,limit)[0]


def verify_source(panel,meta,calendar,arrays,source239,report,limit=None):
    wanted=independent_source(panel,meta,calendar,arrays,source239,limit)
    record_check(report,wanted,'Independent daily source report')
    return dict(**{k:True for k in SOURCE_FLAGS},counts=wanted['counts'],checked_events=wanted['counts']['events'],
      full_source_report_supplied=True,numerical_tolerance=0.,episode_calls=0,new_target_ranks_computed=False,
      candidate_selection_computed=False,producer_modules_imported=False,computes_returns=False)


def _prior_metadata(daily,n):
    require(type(daily) is dict and set(daily)==set(SOURCE_TOP_KEYS),'Daily source report fields')
    require(daily['schema']==SOURCE_SCHEMA and daily['daily_paths_checked'] is True,'Daily source schema')
    prior={k:copy.deepcopy(v) for k,v in daily.items() if k not in ('daily_paths_checked','events','records')}
    prior['schema']='peer239_mature_source_report_v1'
    require(type(daily['events']) is list and len(daily['events'])>=n,'Daily event prefix')
    prior['events']=[{k:copy.deepcopy(e[k]) for k in SOURCE_EVENT_KEYS[:-1]} for e in daily['events'][:n]]
    rows=[]
    require(type(daily['records']) is list,'Daily records')
    for r in daily['records']:
        require(type(r) is dict and type(r.get('query_index')) is int and r['query_index']>=0,'Daily record query')
        if r['query_index']<n:rows.append({k:copy.deepcopy(r[k]) for k in LEGACY_SOURCE_RECORD_KEYS})
    prior['records']=rows
    return prior


def _daily_inputs(panel,meta,calendar,arrays,daily,limit):
    n=peer.axes(meta,arrays,limit)[0]
    prior=_prior_metadata(daily,n)
    expected,paths,axes=_source(panel,meta,calendar,arrays,prior,limit)
    for k in SOURCE_TOP_KEYS:
        if k not in ('events','records','counts'):record_check(daily[k],expected[k],'Daily source/'+k)
    record_check(daily['events'][:n],expected['events'],'Daily events')
    consumed=[r for r in daily['records'] if r['query_index']<n]
    record_check(consumed,expected['records'],'Daily records')
    require(type(daily['counts']) is dict and set(daily['counts'])==set(COUNT_FIELDS),'Daily count keys')
    require(type(daily['counts']['events']) is int and daily['counts']['events']>=n,'Daily count prefix')
    if len(daily['events'])==n:record_check(daily['counts'],expected['counts'],'Daily exact counts')
    return expected,paths,axes


def pair(value):
    require(isinstance(value,Fraction),'Exact rational evidence')
    return dict(numerator=str(value.numerator),denominator=str(value.denominator))


def rational(value):
    if isinstance(value,Fraction):return value
    require(type(value) is dict and set(value)=={'numerator','denominator'},'Canonical rational pair')
    require(type(value['numerator']) is str and type(value['denominator']) is str,'Rational strings')
    try:p=int(value['numerator']);q=int(value['denominator'])
    except (TypeError,ValueError) as exc:raise ValueError('Rational integers') from exc
    require(str(p)==value['numerator'] and str(q)==value['denominator'] and q>0 and math.gcd(p,q)==1,'Reduced canonical rational')
    return Fraction(p,q)


def median_exact(values):
    require(len(values)>0 and all(isinstance(x,Fraction) for x in values),'Nonempty exact median')
    ordered=sorted(values);k=len(ordered)//2
    return ordered[k] if len(ordered)%2 else (ordered[k-1]+ordered[k])/2


def ranks_exact(values):
    require(all(isinstance(x,Fraction) for x in values),'Exact rank values')
    n=len(values)
    return [float(Fraction(2*sum(y<x for y in values)+sum(y==x for y in values)+1-n,2*n)) for x in values]


def _relative(focal,peers):
    require(len(focal)==211 and len(peers)>=2 and all(len(p)==211 for p in peers),'211 prices and two peers')
    focal_gross=focal[-1]/focal[0]
    # Independent route: median of exact SIMPLE returns, then add one.
    daily=Fraction(1)
    for d in range(1,211):daily*=1+median_exact([p[d]/p[d-1]-1 for p in peers])
    endpoint=median_exact([p[-1]/p[0] for p in peers])
    require(daily>0 and endpoint>0,'Positive exact reference path')
    return dict(focal_gross=focal_gross,daily_peer_gross=daily,endpoint_peer_gross=endpoint,
      path_raw=focal_gross/daily-1,endpoint_raw=focal_gross/endpoint-1)


def relative_statistics(focal,peers):
    def path(values):
        require(len(values)==211,'Exactly211 prices');out=[]
        for x in values:
            require(isinstance(x,Real) and not isinstance(x,(bool,np.bool_)),'Real nonboolean source')
            try:f=float(x)
            except (TypeError,ValueError,OverflowError) as exc:raise ValueError('Invalid source conversion') from exc
            require(math.isfinite(f) and f>0,'Positive finite source');out.append(Fraction.from_float(f))
        return out
    return _relative(path(focal),[path(p) for p in peers])


def independent_features(panel,meta,calendar,arrays,daily_source_report,limit=None):
    source,paths,axes=_daily_inputs(panel,meta,calendar,arrays,daily_source_report,limit)
    n,width,days,names,etfs,groups=axes
    values={k:np.zeros((n,width),bool) if k in VALUE_KEYS[:2] else np.full((n,width),np.nan) for k in VALUE_KEYS}
    events=[]
    for i,(day,ss,group) in enumerate(zip(days,names,groups)):
        mature=[r['column_index'] for r in group if r['history_status']=='observed_full_window']
        supported=[r['column_index'] for r in group if r['structural_peer_support']]
        pp={j:[Fraction.from_float(x) for x in paths[i,j]] for j in mature}
        stats={j:_relative(pp[j],[pp[k] for k in next(r for r in group if r['column_index']==j)['peer_indices']]) for j in supported}
        for j in mature:values['mature'][i,j]=True
        for j in supported:values['supported'][i,j]=True
        for key,raw in (('path_target','path_raw'),('endpoint_target','endpoint_raw')):
            for j,q in zip(supported,ranks_exact([stats[j][raw] for j in supported])):values[key][i,j]=q
        cells=[]
        for r in group:
            j=r['column_index'];cell={k:copy.deepcopy(r[k]) for k in CELL_KEYS[:7]}
            for key in CELL_KEYS[7:12]:cell[key]=pair(stats[j][key]) if j in stats else (pair(pp[j][-1]/pp[j][0]) if key=='focal_gross' and j in pp else None)
            for key in VALUE_KEYS[2:]:cell[key]=float(values[key][i,j]) if j in supported else None
            cells.append(cell)
        ev=source['events'][i]
        events.append(dict(query_index=i,date=str(day.date()),source_dates=copy.deepcopy(ev['source_dates']),
          source_positions=copy.deepcopy(ev['source_positions']),stock_indices=copy.deepcopy(ev['stock_indices']),
          mature_indices=mature,supported_indices=supported,cells=cells))
    return values,dict(schema='monthly_peer_path252_features_v2',counts=source['counts'].copy(),events=events)


def verify_features(panel,meta,calendar,arrays,daily_source_report,values,audit,limit=None):
    expected,wanted=independent_features(panel,meta,calendar,arrays,daily_source_report,limit)
    require(type(values) is dict and set(values)==set(VALUE_KEYS),'Exactly four feature arrays')
    for key in VALUE_KEYS:exact_array(values[key],expected[key],'Feature '+key)
    record_check(audit,wanted,'Exact complete feature audit')
    return dict(**{k:True for k in FEATURE_FLAGS},counts=wanted['counts'],events=wanted['counts']['events'],
      checked_events=wanted['counts']['events'],full_features_supplied=True,numerical_tolerance=0.,
      rank_tolerance=0.,exact_discrete=True,producer_modules_imported=False,computes_returns=False)


def selection_evidence(original,meta,values,n):
    _,width,days,names,etfs=peer.axes(meta,original,n)
    require(type(values) is dict and set(values)==set(VALUE_KEYS),'Selection feature keys')
    v={k:np.asarray(values[k])[:n] for k in VALUE_KEYS}
    for k in VALUE_KEYS:
        require(v[k].shape==(n,width) and v[k].dtype==np.dtype(bool if k in VALUE_KEYS[:2] else np.float64),'Feature shape/dtype '+k)
    require(not (v['supported']&~v['mature']).any(),'Support subset mature')
    for k in VALUE_KEYS[2:]:
        require(np.isfinite(v[k][v['supported']]).all() and np.isnan(v[k][~v['supported']]).all(),'Feature availability '+k)
        require((np.abs(v[k][v['supported']])<=.5).all(),'Bounded target '+k)
    for i,(d,ss) in enumerate(zip(days,names)):
        forbidden=[j for j,s in enumerate(ss) if s in etfs or str(d.date())=='2016-08-17']
        require(not v['mature'][i,forbidden].any(),'ETF/partial no new feature')
    return v,width


def _selection(original,meta,values,n=None):
    n=len(meta['dates']) if n is None else n;v,width=selection_evidence(original,meta,values,n)
    base=original_base(original,meta,n);scores={};choices={};pres={};minimum=None;sum_error=0.;maximum=0.
    for arm,key in zip(ARMS,(None,'path_target','endpoint_target')):
        pre=base['P'].copy();delta=np.zeros((n,width));hand=np.zeros((n,width))
        if key is not None:
            good=v['supported'];delta[good]=.05*(v[key][good]-base['support'][good]);hand[good]=.05
            require(np.isfinite(delta).all(),'Finite funded correction');nz=delta!=0.;pre[nz]+=delta[nz]
        if n:maximum=max(maximum,float(np.max(np.abs(delta))))
        pres[arm]=pre.copy();score,rows=old.episode(pre,base)
        scores[arm]=score;choices[arm]=rows
        for i in range(n):
            for j in range(width):
                weights=[float(base['cw'][i]),.025*float(base['calm'][i]),.05*float(base['weak'][i]),
                  .01*float(base['A'][i,j]),float(base['budget'][i]),float(hand[i,j])]
                remaining=1.-weights[0]
                for w in weights[1:]:remaining-=w
                error=abs(math.fsum(weights+[remaining])-1.)
                require(remaining>=.33-1e-12 and error<=1e-12,'Complete original budget including extra.01')
                minimum=remaining if minimum is None else min(minimum,remaining);sum_error=max(sum_error,error)
    return scores,choices,pres,dict(minimum_remaining_support=minimum,maximum_budget_sum_error=sum_error,
      calm_nonweak_extension_count=int(base['A'].sum()),maximum_absolute_delta=maximum)


def selection_oracle(original,meta,values,n=None):
    s,c,p,_=_selection(original,meta,values,n);return s,c,p


def orders_from_choices(scores,choices,meta):
    return {arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
      symbols=[meta['symbols'][i][j] for j in ids],scores=[float(scores[arm][i,j]) for j in ids])
      for i,ids in enumerate(choices[arm])] for arm in ARMS}


def verify_selection(original,meta,values,orders,*,scores,pre_scores,n=None):
    n=len(meta['dates']) if n is None else n;s,c,p,budget=_selection(original,meta,values,n)
    record_check(orders,orders_from_choices(s,c,meta),'Exact ordered selection')
    for got,expected,label in ((scores,s,'Final scores'),(pre_scores,p,'Modified pre')):
        require(type(got) is dict and set(got)==set(ARMS),'Three '+label+' arms')
        for arm in ARMS:exact_array(got[arm],expected[arm],label+'/'+arm)
    return dict(**{k:True for k in SELECTION_FLAGS},**budget,checked_events=n,ordered_choices_checked=3*n,
      full_scores_supplied=True,full_pre_scores_supplied=True,numerical_tolerance=0.,
      original_source_parity_tolerance=1e-12,producer_modules_imported=False,computes_returns=False,
      holdings_counts={a:{str(k):sum(len(r)==k for r in c[a]) for k in (0,2,3)} for a in ARMS})
def audit_outcomes(px,opening,orders,exports,results,meta):
    """Three arms, three paths and five slices; reuse only independent accounting."""
    helper=accounting()
    require(set(orders)==set(exports)==set(results)==set(ARMS),'Outcome arm names differ')
    require(len(meta['dates'])==len(meta['ends'])==122 and meta['ends'][:-1]==meta['dates'][1:],'Outcome interval axis differs')
    require(meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24' and
        meta['ends'][52]==meta['dates'][53]=='2020-12-31' and meta['ends'][87]==meta['dates'][88]=='2023-11-30',
        'Outcome fixed slices differ')
    reports={}
    for arm in ARMS:
        require(len(orders[arm])==122 and [o['date'] for o in orders[arm]]==meta['dates'] and
                [o['end'] for o in orders[arm]]==meta['ends'],'Order axis differs')
        require(set(exports[arm])==set(results[arm])=={p[0] for p in PATHS},'Execution paths differ')
        reports[arm]={}
        for name,execution,cost in PATHS:
            data=exports[arm][name]
            require(set(data)=={'DAILY','TRADES','HOLDINGS','DAILY_POSITIONS'},'Incomplete execution exports')
            if cost==0.:
                require(all(float(row['cost'])==0. for row in data['TRADES']), 'Zero-cost trade fees')
                require(all(float(row['fee'])==0. for row in data['HOLDINGS']), 'Zero-cost period fees')
            audited=helper.audit_path(px,opening,orders[arm],*[data[k] for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')],execution,cost)
            slices=helper.statistics_by_split(audited['days'],audited['values'],meta);wanted=results[arm][name]
            if cost==0.:require(audited['fees_paid']==wanted['metrics']['fees_paid']==0.,'Zero-cost aggregate fees exact zero')
            require(set(wanted['slices'])==set(slices),'Fixed five slices differ')
            helper.check_metrics(slices['full122'],wanted['metrics'],arm+'/'+name)
            for split,value in slices.items():helper.check_metrics(value,wanted['slices'][split],arm+'/'+name+'/'+split)
            for key in ('fees_paid','dollar_turnover','trade_rows'):helper.near(audited[key],wanted['metrics'][key],key)
            annual=[]
            for year in sorted({d[:4] for d in audited['days']}):
                ids=[i for i,day in enumerate(audited['days']) if day.startswith(year)]
                if len(ids)<2:continue
                prior=audited['values'][ids[0]-1] if ids[0] else 100000.
                annual.append(dict(year=int(year),return_=audited['values'][ids[-1]]/prior-1.,partial=int(year) in (2016,2026)))
            # Preserve the frozen metric comparison tolerance; feature/selection checks stay exact.
            require(len(wanted['metrics']['annual_returns'])==len(annual),'Annual metric count')
            for got,expected in zip(wanted['metrics']['annual_returns'],annual):
                require(set(got)==set(expected) and got['year']==expected['year'] and got['partial']==expected['partial'],'Annual metric identity')
                helper.near(got['return_'],expected['return_'],'Annual return')
            require(wanted['accounting']['exported_fill_ledger_matches'] is True and
                    wanted['accounting']['terminal_liquidation'] is False,'Account declaration differs')
            for key in ('days','values','daily_cash_nav'):audited.pop(key)
            reports[arm][name]=dict(audited,passed=True,slices=slices)
    return dict(passed=True,checked_paths=9,checked_slices=45,checked_training_groups=18,independent_accounts=True,
        independent_of_production_model_and_engine=True,terminal_liquidation=False,
        accounting_source_sha256=ACCOUNTING_SHA,paths=reports)
