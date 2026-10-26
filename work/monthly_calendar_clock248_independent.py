"""Independent original-core/calendar-clock and single-episode verification.

Only the pinned independent original-score/accounting bridge is reused.
No new producer/host or raw-source loader is imported.
"""
import copy
from datetime import date
import hashlib
import importlib.util
import math
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'work/monthly_core_veto241_independent.py'
PRIOR_SHA='642958e877a0ed53347c2ca21aea0027e0148591af21985268976b91d9e22fb6'
PRIOR_TEST=ROOT/'work/test_monthly_core_veto241_independent.py'
PRIOR_TEST_SHA='4fc17fdcf385f30a87baf66513a1b2e0c19db2e0549dc779b6fe41f755e64087'
ACCOUNT_TEST=ROOT/'work/test_monthly_upside_retention181_independent.py'
ACCOUNT_TEST_SHA='b597ad5d7d128625e9d5c698c007cc3a5deafc398acd9ebaf3fb8bd83df23fe0'
ARMS=('incumbent_local67','calendar_mixed_core')
COUNT_FIELDS=('events','symbols_per_event','initial_partial_events','formal_events','calm_formal_events',
 'noncalm_formal_events','original_checked_cells','original_finite_cells','calendar_required_cells',
 'calendar_available_cells','calendar_missing_cells','available_events','empty_domain_events',
 'missing_calendar_events','candidate_changed_core_cells','changed_holdings_events','original_episode_calls',
 'new_model_fits','predictive_model_fits')
EVENT_KEYS=('query_index','date','initial_partial','calm','core_weight','symbols','branch','panel_first_session',
 'query_position','next_official_session','trading_price_start','trading_price_end','risk_price_start','risk_price_end',
 'calendar_start','calendar_end','calendar_lag_sessions','u0_indices','calendar_missing_indices',
 'core_source_max_abs_error','cells')
CELL_KEYS=('column_index','symbol','original_finite','original_raw','original_rank','frozen_core','rebuilt_core',
 'core_abs_error','sigma63','calendar_source_available','calendar_reason','calendar_start_price',
 'calendar_end_price','calendar_raw','calendar_rank','delta')
DECISION_KEYS=('query_index','date','branch','u0_indices','candidate_delta','extension_active_indices',
 'episode_budget','original_remaining_support','incumbent_choices','candidate_choices')
SELECTION_FLAGS=('passed','independent_original_pre_checked','original_model_rank_scope_preserved',
 'original_eligibility_unchanged','original_support_checked','original_budget_checked',
 'calm_nonweak_extension_budget_checked','original_effects_unchanged','original_core_checked',
 'calendar_anchors_checked','same_original_finite_domain_checked','pre_eligibility_ranking_checked',
 'original_etf_domain_preserved','average_percentile_ties_checked','whole_month_fallback_checked',
 'full_core_delta_checked','frozen_core_delta_anchor_checked','single_original_episode',
 'no_second_episode','named_initial_partial_unchanged','exact_incumbent_copy_checked',
 'exact_scores_and_choices_checked','full_audit_checked','descriptive_counts_checked','no_predictive_fits_checked')
PREPARE_FLAGS=SELECTION_FLAGS+('independent_source_rebuild','original_full_history_pandas_checked',
 'original_official_row_clock_checked','original_raw_rank_parity_checked','original_weighted_core_tolerance_checked',
 'all_U0_endpoints_checked','source_missing_not_imputed','invalid_source_not_hidden_by_missing',
 'visible_listing_bounds_checked','continuous_statistics_checked','independent_rank_agreement_checked',
 'no_future_source_values')
SOURCE_GATE_FLAGS=('passed','full_features_supplied','original_core_checked','calendar_anchors_checked',
 'original_raw_rank_parity_checked','same_original_finite_domain_checked','whole_month_fallback_checked',
 'visible_listing_bounds_checked','exact_discrete')
INCUMBENT_GATE_FLAGS=('passed','full_scores_supplied','full_pre_scores_supplied','independent_original_pre_checked',
 'exact_incumbent_copy_checked','original_eligibility_unchanged','single_original_episode')
CONTINUOUS_ATOL=CONTINUOUS_RTOL=2e-10
C_SOURCE_ATOL=1e-12
C_SOURCE_RTOL=0.


def load_pinned(path,digest,name):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Frozen independent dependency changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


old=load_pinned(PRIOR,PRIOR_SHA,'calendar248_original_independent')
require,exact_array,record_check=old.require,old.exact_array,old.record_check
PATHS,ACCOUNTING_SHA=old.PATHS,old.ACCOUNTING_SHA

def accounting():return old.accounting()
def frozen_source_oracle():return old.frozen_source_oracle()
def dependency_files():
    for p,h in ((PRIOR_TEST,PRIOR_TEST_SHA),(ACCOUNT_TEST,ACCOUNT_TEST_SHA)):
        require(hashlib.sha256(p.read_bytes()).hexdigest()==h,'Frozen synthetic fixture changed')
    return tuple(sorted(set(old.dependency_files())|{PRIOR,PRIOR_TEST,ACCOUNT_TEST,Path(__file__).resolve()}))


def iso(value):
    require(type(value) is str,'Canonical ISO date');day=date.fromisoformat(value)
    require(day.isoformat()==value,'Canonical ISO date');return pd.Timestamp(day)


def finite(value,label):
    value=float(value);require(math.isfinite(value),'Nonfinite '+label);return value


def source_axes(panel,calendar,n):
    require(type(panel) is dict and set(panel)=={'adj'},'Adjusted-only source panel')
    frame=panel['adj'];require(isinstance(frame,pd.DataFrame) and isinstance(frame.index,pd.DatetimeIndex),'Price frame')
    ix=frame.index
    require(ix.is_unique and ix.is_monotonic_increasing and not ix.hasnans and ix.tz is None and ix.equals(ix.normalize()),'Price date axis')
    require(frame.columns.is_unique and all(type(s) is str and s and s.strip()==s for s in frame.columns),'Price columns')
    require(isinstance(calendar,(list,tuple)) and (calendar or n==0),'Official calendar')
    sessions=[iso(s) for s in calendar]
    require(all(d.weekday()<5 for d in sessions) and all(a<b for a,b in zip(sessions,sessions[1:])),'Official ordered weekdays')
    return frame,sessions


def independent_rank(raw):
    """Pair-count rank, separately checked against the original pandas convention."""
    valid=[j for j,v in enumerate(raw) if math.isfinite(float(v))];out=np.zeros(len(raw),float)
    for j in valid:
        less=sum(raw[k]<raw[j] for k in valid);equal=sum(raw[k]==raw[j] for k in valid)
        out[j]=(less+(equal+1)/2.)/len(valid)-.5
    reference=pd.Series(raw,dtype=float).replace([np.inf,-np.inf],np.nan).rank(pct=True).fillna(.5).to_numpy()-.5
    exact_array(out,reference,'Independent original percentile ranks')
    return out,valid


def original_values(frame,day,names,first_session):
    """Keep the full original pandas past, not a reset last64 rolling window."""
    prefix=frame.loc[:day,names]
    require(len(prefix)>0 and str(prefix.index[0].date())==first_session,'Full original panel origin')
    out=[];vols=[]
    for symbol in names:
        converted=[]
        for value in prefix[symbol].tolist():
            if value is None or value is pd.NA:
                converted.append(float('nan'));continue
            require(isinstance(value,Real) and not isinstance(value,(bool,np.bool_)),'Original real nonboolean history')
            try:converted.append(float(value))
            except (TypeError,ValueError,OverflowError) as e:raise ValueError('Unrepresentable original source') from e
        series=pd.Series(converted,index=prefix.index,dtype=float)
        returns=series.pct_change(fill_method=None)
        sigma=returns.rolling(63).std()
        vols.append(float(sigma.iloc[-1]))
        raw210=series.shift(42)/series.shift(252)-1.
        raw252=(series.shift(42)/series.shift(294)-1.)/sigma.replace(0.,np.nan)
        out.append((float(raw210.iloc[-1]),float(raw252.iloc[-1])))
    return out,vols


def original_base(original,meta,n):
    require(type(n) is int and 0<=n<=len(meta['dates']),'Consumed prefix')
    width,days,names=old.axes(original,meta,n);require(width>0,'Original width positive')
    if not n:return dict(width=width,days=days,names=names,P=np.empty((0,width)),E=np.empty((0,width),bool),
        core=np.empty((0,width)),cw=np.empty(0),calm=np.empty(0,bool),weak=np.empty(0,bool),
        support=np.empty((0,width)),effects=np.empty((0,width)),A=np.empty((0,width),bool),budget=np.empty(0))
    x=frozen_source_oracle().old.original_helper().original_prefix(original,meta,n)
    cw=np.where(x['calm'],.5,.52)
    exact_array(x['cw'],np.broadcast_to(cw[:,None],(n,width)),'Original core weights')
    ext=np.asarray(original['extension_available'])[:n];fore=np.asarray(original['forecast_available'])[:n]
    ranklog=np.asarray(original['rank_log_excess'])[:n]
    require(ext.shape==fore.shape==ranklog.shape==(n,width) and ext.dtype==fore.dtype==np.dtype(bool),'Original extension axes')
    A=(x['calm']&~x['weak'])[:,None]&ext&fore&x['eligible']&np.isfinite(ranklog)
    return dict(width=width,days=days,names=names,P=x['local'].copy(),E=x['eligible'],core=x['core'],cw=cw,
        calm=x['calm'],weak=x['weak'],support=x['support'],effects=x['effects'],A=A,
        budget=np.minimum(.05,np.maximum(0.,.10-(.025*x['calm']+.05*x['weak']))))


def calendar_anchors(day,sessions):
    period=day.to_period('M');answer={}
    for lag in (2,12,14):
        pool=[d for d in sessions if d.to_period('M')==period-lag]
        require(pool,'Required official calendar anchor month')
        answer['M-'+str(lag)]=str(pool[-1].date())
    return answer


def event_clock(i,base,meta,sessions,origin):
    day=base['days'][i];require(day in sessions,'Query official session');t=sessions.index(day)
    require(t>=294 and t+1<len(sessions),'Original official history plus next witness')
    partial=meta['dates'][i]=='2016-08-17';calm=bool(base['calm'][i])
    if not partial:require(sessions[t+1].to_period('M')!=day.to_period('M'),'Official month-end')
    anchors=calendar_anchors(day,sessions) if not partial else None
    start=str(sessions[t-(252 if calm else 294)].date());end=str(sessions[t-42].date())
    return dict(query_index=i,date=meta['dates'][i],initial_partial=partial,calm=calm,core_weight=float(base['cw'][i]),
        symbols=list(base['names'][i]),branch='initial_partial' if partial else None,panel_first_session=origin,
        query_position=t,next_official_session=str(sessions[t+1].date()),trading_price_start=start,trading_price_end=end,
        risk_price_start=str(sessions[t-63].date()),risk_price_end=meta['dates'][i],
        calendar_start=None if partial else anchors['M-12' if calm else 'M-14'],
        calendar_end=None if partial else anchors['M-2'],
        calendar_lag_sessions=None if partial else t-sessions.index(iso(anchors['M-2'])),
        u0_indices=[],calendar_missing_indices=[],core_source_max_abs_error=None,cells=[])


def review_index(review,n):
    require(isinstance(review,dict) and review.get('schema')=='calendar_clock248_metadata_source_review_v1','Source review schema')
    origin=review.get('panel_first_session');iso(origin)
    require(isinstance(review.get('listing_bounds'),list) and isinstance(review.get('identity_cells'),list),'Source identity lists')
    bounds={}
    for item in review['listing_bounds']:
        require(type(item) is dict and set(item)=={'symbol','issuer_key','not_before','available_from','kind','evidence_id'},'Listing bound fields')
        require(all(type(v) is str and v and v.strip()==v for v in item.values()),'Listing bound strings')
        iso(item['not_before']);iso(item['available_from']);require(item['symbol'] not in bounds,'Unique listed security')
        bounds[item['symbol']]=item
    identities={}
    for item in review['identity_cells']:
        require(type(item) is dict and type(item.get('query_index')) is int and item['query_index']>=0,'Identity query index')
        q=item['query_index']
        if q>=n:continue
        j=item.get('column_index');require(type(j) is int and j>=0,'Identity column')
        require((q,j) not in identities,'Duplicate listing identity');identities[q,j]=item
    return origin,bounds,identities


def listing_checks(event,sessions,bounds,identities):
    found={};q=event['query_index'];t=event['query_position']
    for j,s in enumerate(event['symbols']):
        if s not in bounds:continue
        b=bounds[s];item=identities.get((q,j));require(item is not None,'Missing listing occurrence')
        wanted=dict(query_index=q,column_index=j,date=event['date'],symbol=s,issuer_key=b['issuer_key'],
            available_from=b['available_from'],not_before=b['not_before'],bound_visible=b['available_from']<=event['date'],
            calendar_anchors=calendar_anchors(iso(event['date']),sessions),
            old_clock_on_official_axis=dict(t_minus_42=str(sessions[t-42].date()),t_minus_252=str(sessions[t-252].date()),
                t_minus_294=str(sessions[t-294].date()),risk63_price_start=str(sessions[t-63].date()),
                risk63_price_end=event['date'],risk63_prices=64,risk63_return_intervals=63))
        for k,v in wanted.items():record_check(item.get(k),v,'Listing '+k)
        found[j]=b
    require({j for (i,j) in identities if i==q}==set(found),'No stray listing occurrence')
    return found


def calendar_source(frame,symbol,start,end,bound,query):
    """Scan both endpoints; missing cannot hide an invalid or pre-inception value."""
    prices=[];missing=[]
    for day in (start,end):
        if symbol not in frame.columns:missing.append('missing_symbol');prices.append(None);continue
        if iso(day) not in frame.index:missing.append('missing_row');prices.append(None);continue
        value=frame.at[iso(day),symbol]
        if value is None or value is pd.NA:missing.append('missing_nan');prices.append(None);continue
        require(isinstance(value,Real) and not isinstance(value,(bool,np.bool_)),'Calendar real nonboolean endpoint')
        try:value=float(value)
        except (TypeError,ValueError,OverflowError) as e:raise ValueError('Unrepresentable calendar endpoint') from e
        if math.isnan(value):missing.append('missing_nan');prices.append(None);continue
        require(math.isfinite(value) and value>0.,'Positive finite calendar endpoint')
        if bound and bound['available_from']<=query:require(day>=bound['not_before'],'Calendar source predates visible listing bound')
        prices.append(value)
    if missing:
        reason=min(missing,key=('missing_symbol','missing_row','missing_nan').index)
        return False,reason,prices[0],prices[1]
    return True,'available',prices[0],prices[1]


def independent_features(panel,meta,calendar,original,review,n=None):
    n=len(meta['dates']) if n is None else n;base=original_base(original,meta,n)
    frame,sessions=source_axes(panel,calendar,n);origin,bounds,identities=review_index(review,n)
    features=[]
    for i in range(n):
        e=event_clock(i,base,meta,sessions,origin);bound=listing_checks(e,sessions,bounds,identities)
        if e['initial_partial']:features.append(e);continue
        day=base['days'][i];t=e['query_position'];names=e['symbols']
        require(all(s in frame.columns for s in names),'All original source columns required')
        prefix=frame.loc[:day]
        require(len(prefix)>=295 and prefix.index[-295:].equals(pd.DatetimeIndex(sessions[t-294:t+1])),
            'Original last295 rows exactly official sessions')
        pairs,vol=original_values(frame,day,names,origin)
        raw=np.array([p[0 if e['calm'] else 1] for p in pairs]);raw[~np.isfinite(raw)]=np.nan
        ranks,U=independent_rank(raw);e['u0_indices']=U
        for j,s in enumerate(names):
            rebuilt=float(e['core_weight']*ranks[j]);frozen=float(base['core'][i,j]);err=abs(rebuilt-frozen)
            require(err<=C_SOURCE_ATOL,'Original weighted-core source parity')
            valid=j in U;b=bound.get(j)
            if valid and b and b['available_from']<=e['date']:
                require(e['trading_price_start']>=b['not_before'],'Original finite core predates visible listing bound')
            e['cells'].append(dict(column_index=j,symbol=s,original_finite=valid,
                original_raw=float(raw[j]) if valid else None,original_rank=float(ranks[j]),frozen_core=frozen,
                rebuilt_core=rebuilt,core_abs_error=err,sigma63=float(vol[j]) if math.isfinite(vol[j]) else None,
                calendar_source_available=False,calendar_reason='outside_U0',calendar_start_price=None,
                calendar_end_price=None,calendar_raw=None,calendar_rank=None,delta=0.))
        e['core_source_max_abs_error']=max(c['core_abs_error'] for c in e['cells'])
        for j in U:
            c=e['cells'][j];ok,reason,p0,p1=calendar_source(frame,c['symbol'],e['calendar_start'],e['calendar_end'],bound.get(j),e['date'])
            c.update(calendar_source_available=ok,calendar_reason=reason,calendar_start_price=p0,calendar_end_price=p1)
            if not ok:e['calendar_missing_indices'].append(j);continue
            value=finite(p1/p0-1.,'Calendar raw return')
            if not e['calm']:
                require(c['sigma63'] is not None and c['sigma63']>0.,'Original positive denominator on U0')
                value=finite(value/c['sigma63'],'Calendar risk-adjusted raw')
            c['calendar_raw']=value
        if not U:e['branch']='empty_domain'
        elif e['calendar_missing_indices']:e['branch']='missing_calendar'
        else:
            e['branch']='available';vector=[c['calendar_raw'] if c['original_finite'] else np.nan for c in e['cells']]
            newr,newU=independent_rank(vector);require(newU==U,'Same finite ranking domain')
            for j in U:
                c=e['cells'][j];c['calendar_rank']=float(newr[j]);c['delta']=finite(e['core_weight']*newr[j]-c['frozen_core'],'Core delta')
        features.append(e)
    return features


def exact_indices(value,width,label):
    require(type(value) is list and all(type(j) is int and 0<=j<width for j in value) and value==sorted(set(value)),label)


def validated_deltas(features,base,meta,n):
    require(type(features) is list and len(features)>=n,'Complete feature prefix');width=base['width'];deltas=np.zeros((n,width))
    for i,e in enumerate(features[:n]):
        require(type(e) is dict and set(e)==set(EVENT_KEYS),'Feature event schema')
        for k,v in dict(query_index=i,date=meta['dates'][i],initial_partial=meta['dates'][i]=='2016-08-17',
                        calm=bool(base['calm'][i]),core_weight=float(base['cw'][i]),symbols=list(base['names'][i])).items():
            record_check(e[k],v,'Cached '+k)
        iso(e['panel_first_session']);require(type(e['query_position']) is int and e['query_position']>=294,'Cached official position')
        require(iso(e['next_official_session'])>iso(e['date']),'Cached next session')
        for k in ('trading_price_start','trading_price_end','risk_price_start','risk_price_end'):iso(e[k])
        require(e['risk_price_end']==e['date'],'Cached current risk end')
        if e['initial_partial']:
            require(e['branch']=='initial_partial' and e['cells']==[] and e['u0_indices']==[] and e['calendar_missing_indices']==[]
                and e['core_source_max_abs_error'] is None and all(e[k] is None for k in ('calendar_start','calendar_end','calendar_lag_sessions')),
                'Partial exact bypass');continue
        require(iso(e['calendar_start'])<iso(e['calendar_end'])<iso(e['date']),'Cached calendar ordering')
        month=iso(e['date']).to_period('M')
        require(iso(e['calendar_start']).to_period('M')==month-(12 if e['calm'] else 14) and
                iso(e['calendar_end']).to_period('M')==month-2,'Cached calendar month anchors')
        require(type(e['calendar_lag_sessions']) is int and e['calendar_lag_sessions']>0,'Cached calendar session gap')
        require(type(e['cells']) is list and len(e['cells'])==width,'All original cells')
        U=[];missing=[];raw=[];newraw=[]
        for j,c in enumerate(e['cells']):
            require(type(c) is dict and set(c)==set(CELL_KEYS),'Feature cell schema')
            record_check(c['column_index'],j,'Column');record_check(c['symbol'],base['names'][i][j],'Symbol')
            require(type(c['original_finite']) is bool and type(c['calendar_source_available']) is bool,'Cell boolean')
            record_check(c['frozen_core'],float(base['core'][i,j]),'Frozen core anchor')
            if c['original_finite']:
                raw.append(finite(c['original_raw'],'Original raw'));U.append(j)
            else:require(c['original_raw'] is None,'Original missing raw');raw.append(np.nan)
            if c['sigma63'] is not None:require(finite(c['sigma63'],'Original sigma')>=0.,'Nonnegative original sigma')
            if not c['original_finite']:
                require(c['calendar_source_available'] is False and c['calendar_reason']=='outside_U0' and
                    all(c[k] is None for k in ('calendar_start_price','calendar_end_price','calendar_raw','calendar_rank')) and
                    c['delta']==0.,'No outside-domain calendar input');newraw.append(np.nan);continue
            if c['calendar_source_available']:
                require(c['calendar_reason']=='available','Available source reason')
                p0=finite(c['calendar_start_price'],'Calendar start');p1=finite(c['calendar_end_price'],'Calendar end')
                require(p0>0. and p1>0.,'Positive cached endpoints');v=finite(p1/p0-1.,'Cached calendar raw')
                if not e['calm']:
                    require(c['sigma63'] is not None and c['sigma63']>0.,'Positive cached sigma');v=finite(v/c['sigma63'],'Cached risk raw')
                record_check(c['calendar_raw'],v,'Cached canonical calendar raw');newraw.append(v)
            else:
                require(c['calendar_reason'] in ('missing_symbol','missing_row','missing_nan') and
                    all(c[k] is None for k in ('calendar_raw','calendar_rank')) and
                    (c['calendar_start_price'] is None or c['calendar_end_price'] is None),'Missing source evidence')
                for pk in ('calendar_start_price','calendar_end_price'):
                    if c[pk] is not None:require(finite(c[pk],'Partial cached calendar endpoint')>0.,'Positive partial endpoint')
                newraw.append(np.nan);missing.append(j)
        ranks,rankU=independent_rank(raw);require(rankU==U,'Cached source finite domain')
        exact_indices(e['u0_indices'],width,'U0 sorted exact indices');record_check(e['u0_indices'],U,'Cached U0')
        exact_indices(e['calendar_missing_indices'],width,'Missing sorted exact indices');record_check(e['calendar_missing_indices'],missing,'Cached missing domain')
        errors=[]
        for j,c in enumerate(e['cells']):
            rebuilt=float(e['core_weight']*ranks[j]);err=abs(rebuilt-float(base['core'][i,j]));require(err<=C_SOURCE_ATOL,'Cached old C parity')
            for k,v in (('original_rank',float(ranks[j])),('rebuilt_core',rebuilt),('core_abs_error',err)):
                record_check(c[k],v,'Cached '+k)
            errors.append(err)
        record_check(e['core_source_max_abs_error'],max(errors),'Cached core max error')
        branch='empty_domain' if not U else 'missing_calendar' if missing else 'available'
        require(e['branch']==branch,'Common fallback branch')
        if branch=='available':newr,newU=independent_rank(newraw);require(newU==U,'No reference-domain change')
        for j,c in enumerate(e['cells']):
            wanted_rank=float(newr[j]) if branch=='available' and j in U else None
            record_check(c['calendar_rank'],wanted_rank,'Cached calendar rank')
            d=float(e['core_weight']*wanted_rank-c['frozen_core']) if wanted_rank is not None else 0.
            record_check(c['delta'],d,'Cached core replacement');deltas[i,j]=d
    return deltas


def episode(pre,base):
    scores=pre.copy();choices=[]
    for i,ss in enumerate(base['names']):
        rank=sorted(np.flatnonzero(base['E'][i]).tolist(),key=lambda j:(-pre[i,j],ss[j]))
        if len(rank)<3:choices.append(rank if len(rank)==2 else []);continue
        J=[j for j in rank[2:5] if pre[i,rank[2]]-pre[i,j]<=.02]
        for j in J:
            target=min(.5,max(-.5,base['support'][i,j]+base['effects'][i,j]))
            scores[i,j]+=base['budget'][i]*(target-base['support'][i,j])
        winner=min(J,key=lambda j:(-scores[i,j],ss[j]))
        choices.append(sorted(rank[:2]+[winner],key=lambda j:(-scores[i,j],ss[j])))
    require(np.isfinite(scores).all(),'Finite independent episode scores');return scores,choices


def selection_oracle(original,meta,features,n=None):
    n=len(meta['dates']) if n is None else n;base=original_base(original,meta,n)
    deltas=validated_deltas(features,base,meta,n)
    pres={arm:base['P'].copy() for arm in ARMS};changed=deltas!=0.;pres[ARMS[1]][changed]+=deltas[changed]
    scores={};choices={}
    for arm in ARMS:scores[arm],choices[arm]=episode(pres[arm],base)
    counts={k:0 for k in COUNT_FIELDS};counts.update(events=n,symbols_per_event=base['width'],original_episode_calls=2)
    events=[]
    for i,e in enumerate(features[:n]):
        partial=e['initial_partial'];cells=e['cells'];formal=not partial
        counts['initial_partial_events']+=int(partial);counts['formal_events']+=int(formal)
        counts['calm_formal_events']+=int(formal and e['calm']);counts['noncalm_formal_events']+=int(formal and not e['calm'])
        counts['original_checked_cells']+=len(cells);counts['original_finite_cells']+=len(e['u0_indices'])
        counts['calendar_required_cells']+=len(e['u0_indices']);counts['calendar_available_cells']+=sum(c['calendar_source_available'] for c in cells)
        counts['calendar_missing_cells']+=len(e['calendar_missing_indices'])
        for branch,key in (('available','available_events'),('empty_domain','empty_domain_events'),('missing_calendar','missing_calendar_events')):
            counts[key]+=int(e['branch']==branch)
        counts['candidate_changed_core_cells']+=int(np.count_nonzero(deltas[i]))
        counts['changed_holdings_events']+=int(set(choices[ARMS[0]][i])!=set(choices[ARMS[1]][i]))
        remaining=(1.-base['cw'][i])-(.025*base['calm'][i]+.05*base['weak'][i]+.01*base['A'][i])-base['budget'][i]
        require(np.min(remaining)>=.38-1e-12,'Original support budget unchanged')
        events.append(dict(query_index=i,date=meta['dates'][i],branch=e['branch'],u0_indices=e['u0_indices'].copy(),
            candidate_delta=deltas[i].tolist(),extension_active_indices=np.flatnonzero(base['A'][i]).tolist(),
            episode_budget=float(base['budget'][i]),original_remaining_support=remaining.tolist(),
            incumbent_choices=choices[ARMS[0]][i],candidate_choices=choices[ARMS[1]][i]))
    return scores,choices,pres,dict(schema='monthly_calendar_clock248_decisions_v1',counts=counts,events=events)


def orders_from_choices(scores,choices,meta):
    return {arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
        symbols=[meta['symbols'][i][j] for j in ids],scores=[float(scores[arm][i,j]) for j in ids])
        for i,ids in enumerate(choices[arm])] for arm in ARMS}


def verify_selection(original,meta,features,orders,*,scores,pre_scores,audit,n=None):
    n=len(meta['dates']) if n is None else n
    expected,choices,pres,wanted=selection_oracle(original,meta,features,n)
    record_check(orders,orders_from_choices(expected,choices,meta),'Independent ordered exports')
    for got,ref,label in ((scores,expected,'Final scores'),(pre_scores,pres,'Pre scores')):
        require(type(got) is dict and set(got)==set(ARMS),label+' arms')
        for arm in ARMS:exact_array(got[arm],ref[arm],label+'/'+arm)
    record_check(audit,wanted,'Independent complete audit')
    return dict(**{k:True for k in SELECTION_FLAGS},counts=wanted['counts'],checked_events=n,ordered_choices_checked=2*n,
        producer_modules_imported=False,computes_returns=False,full_features_supplied=True,full_scores_supplied=True,
        full_pre_scores_supplied=True,full_audit_supplied=True,numerical_tolerance=0.,
        original_source_parity_tolerance=C_SOURCE_ATOL,C_source_atol=C_SOURCE_ATOL,C_source_rtol=C_SOURCE_RTOL)


def continuous_check(actual,expected,label):
    require(type(actual) in (float,int) and not isinstance(actual,bool) and math.isfinite(float(actual)),label+' finite numeric')
    require(abs(float(actual)-float(expected))<=CONTINUOUS_ATOL+CONTINUOUS_RTOL*abs(float(expected)),label+' continuous agreement')


def _checked_source_features(panel,meta,calendar,original,review,features,n):
    expected=independent_features(panel,meta,calendar,original,review,n)
    require(type(features) is list and len(features)>=n,'Complete supplied features')
    continuous={'original_raw','sigma63','calendar_raw'}
    for i,(a,e) in enumerate(zip(features[:n],expected)):
        require(type(a) is dict and set(a)==set(e),'Feature event fields')
        for k in e:
            if k!='cells':record_check(a[k],e[k],'Feature/'+str(i)+'/'+k)
        require(type(a['cells']) is list and len(a['cells'])==len(e['cells']),'Feature cell count')
        for j,(ac,ec) in enumerate(zip(a['cells'],e['cells'])):
            require(type(ac) is dict and set(ac)==set(ec),'Feature cell fields')
            for k,v in ec.items():
                if k in continuous and v is not None:continuous_check(ac[k],v,'Cell/'+str(i)+'/'+str(j)+'/'+k)
                else:record_check(ac[k],v,'Cell/'+str(i)+'/'+str(j)+'/'+k)
    return expected


def verify_source_features(panel,meta,calendar,original,review,features,n=None):
    """Pre-decision source gate: no candidate selection or episode call."""
    n=len(meta['dates']) if n is None else n
    _checked_source_features(panel,meta,calendar,original,review,features,n)
    return dict(passed=True,checked_events=n,full_features_supplied=True,original_core_checked=True,
        calendar_anchors_checked=True,original_raw_rank_parity_checked=True,same_original_finite_domain_checked=True,
        whole_month_fallback_checked=True,visible_listing_bounds_checked=True,
        C_source_atol=C_SOURCE_ATOL,C_source_rtol=C_SOURCE_RTOL,
        continuous_atol=CONTINUOUS_ATOL,continuous_rtol=CONTINUOUS_RTOL,rank_tolerance=0.,exact_discrete=True,
        producer_modules_imported=False,computes_returns=False,candidate_selection_computed=False,
        episode_calls=0)


def verify_incumbent(original,meta,orders,scores,pre_scores,n=None):
    """Verify the one original arm before the producer starts its candidate arm."""
    n=len(meta['dates']) if n is None else n;base=original_base(original,meta,n)
    expected,choices=episode(base['P'],base)
    wanted=[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
        symbols=[meta['symbols'][i][j] for j in ids],scores=[float(expected[i,j]) for j in ids])
        for i,ids in enumerate(choices)]
    record_check(orders,wanted,'Pre-candidate original ordered choices')
    exact_array(scores,expected,'Pre-candidate original final scores')
    exact_array(pre_scores,base['P'],'Pre-candidate original pre-scores')
    return dict(passed=True,checked_events=n,ordered_choices_checked=n,full_scores_supplied=True,
        full_pre_scores_supplied=True,independent_original_pre_checked=True,exact_incumbent_copy_checked=True,
        original_eligibility_unchanged=True,single_original_episode=True,numerical_tolerance=0.,
        producer_modules_imported=False,computes_returns=False,candidate_selection_computed=False)


def verify_prepared(panel,meta,calendar,original,review,features,orders,*,scores,pre_scores,audit,n=None):
    n=len(meta['dates']) if n is None else n
    expected=_checked_source_features(panel,meta,calendar,original,review,features,n)
    receipt=verify_selection(original,meta,features,orders,scores=scores,pre_scores=pre_scores,audit=audit,n=n)
    es,ec,ep,ea=selection_oracle(original,meta,expected,n)
    record_check(orders,orders_from_choices(es,ec,meta),'Raw-source independent choices')
    for a in ARMS:
        exact_array(scores[a],es[a],'Raw-source final scores');exact_array(pre_scores[a],ep[a],'Raw-source pre')
    record_check(audit,ea,'Raw-source complete audit')
    receipt.update({k:True for k in PREPARE_FLAGS})
    receipt.update(continuous_atol=CONTINUOUS_ATOL,continuous_rtol=CONTINUOUS_RTOL,rank_tolerance=0.,exact_discrete=True)
    return receipt

def audit_outcomes(px,opening,orders,exports,results,meta):
    """Two arms, three paths and five slices; reuse only independent accounting."""
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
    return dict(passed=True,checked_paths=6,checked_slices=30,checked_training_groups=12,independent_accounts=True,
        independent_of_production_model_and_engine=True,terminal_liquidation=False,
        accounting_source_sha256=ACCOUNTING_SHA,paths=reports)
