"""Independent fixed protocol and scalar event oracle for round148.

No producer modules, market reads or computations occur during import.
"""
import math
from collections import Counter
import numpy as np
import pandas as pd

CONCEPT = 'work/monthly_session_cluster148_concept.json'
CONCEPT_SHA = '1998f70944d1679ca29427825f284d1a08e0f0b0767767af9babd76a2fc9d8ee'
NAMES = ('incumbent_local67','event_frequency_control','low_event_clustering')
HANDS = ('frequency_rank','cluster_rank')
EXTRA = ('frequency','cluster',*HANDS,'available')
METHOD = 'adjusted_session_event_lag1_covariance_skip42'
PATHS = (('close',0.,'same_close'),('open',0.,'next_open_zero'),('open',.0005,'next_open_cost'))
PRICE = 'outputs/monthly_2016_extension/PRICES.npz'
PRICE_SHA = 'c877cbe443964d53972114bc654c473608362a6bb31577fb1de5e00cbd3c1485'
SESSIONS = 'outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'
SESSIONS_SHA = 'c05c15f19bc1f7a3428e456a71199d64f05799ef3c432fa45f7bc485d0ff3cf4'
CALENDAR = 'outputs/monthly_forward_guidance_preflight_v1/CALENDAR_PLAN.json'
CALENDAR_SHA = '4b8f32ad8a555d6537d7cf3bd0f831e64fbb2de26d033c8daaf68590fcd4a6ea'


def require(ok, message):
    if not ok: raise ValueError(message)


def exact(a,b):
    require(type(a) is type(b),'Contract type')
    if isinstance(a,dict):
        require(set(a)==set(b),'Contract keys')
        for k in a: exact(a[k],b[k])
    elif isinstance(a,list):
        require(len(a)==len(b),'Contract length')
        for x,y in zip(a,b):exact(x,y)
    else:require(a==b,'Contract value')


def proposal_contract():
    return dict(total_months=122,training_months=88,early_months=53,recent_months=35,
        later_months=34,walkforward=[[36,60],[60,76],[76,88]],
        execution_paths=[dict(execution=e,cost=c,key=k) for e,c,k in PATHS],
        same_close_cagr='strict_greater_than_both_controls',
        next_open_cost_cagr='noninferior_incumbent_strict_greater_than_frequency_control',
        strict_cagr_epsilon=1e-12,max_sharpe_degradation=.05,max_drawdown_degradation=.03,
        max_early_cagr_degradation=.025,max_recent_cagr_degradation=.025,
        new_model_fits=0,intramonth_trades=False,later34_for_selection=False)


def feature_contract():
    return dict(method=METHOD,event_sessions=210,adjacent_pairs=209,skip_sessions=42,
        first_event_offset=-251,last_event_offset=-42,first_close_offset=-252,
        sign_zero_tolerance=1e-12,event='overnight>1e-12 and intraday< -1e-12',
        covariance='(N11*209-Nleft*Nright)/(209*209)',
        rank='0.5-average_zero_based_rank/(n_valid-1)',
        rank_population='common_observed_original49_including_ETFs_before_trading_eligibility',
        missing_rank=0.,singleton_rank=0.,all_equal_rank=0.,max_abs_rank=.5,dose=.01,
        max_abs_correction=.005,
        apply_to='all_available_before_one_original_episode_selector_and_its_original_eligibility',
        required_prices='211_adjusted_closes_and210_adjusted_opens',
        missing_window='invalidate_both_features_without_dropping_pairs_or_filling')


def bound_ref(reader,ref,pins,message):
    require(type(ref) is dict and set(ref)=={'path','sha256'},message+' reference')
    path=reader.path(ref['path'])
    require(reader.sha(path)==ref['sha256']==pins.get(str(path)),message+' digest')
    return path


def verify_proposal(reader,plan,pins):
    p=reader.json(bound_ref(reader,plan['inputs']['proposal'],pins,'Proposal'))
    require(p.get('round')==148 and p.get('arms')==list(NAMES)
        and p.get('primary_challenger')==NAMES[2] and p.get('dose')==.01,'Fixed148 arms')
    exact(p.get('selection_contract'),proposal_contract())
    exact(p.get('feature_contract'),feature_contract())
    exact(p.get('concept'),dict(path=CONCEPT,sha256=CONCEPT_SHA))
    reader.json(bound_ref(reader,p['concept'],pins,'Concept'))
    for k in ('actual_new_factors','actual_new_orders','actual_new_returns'):
        require(type(p.get(k)) is int and p[k]==0,'Prospective calculations')
    fixed=p.get('fixed',{})
    for k,v in dict(signal_months=122,training_months=88,width=49,original_arrays=27,
                    new_downloads=0,new_model_fits=0,observer_changes=0).items():
        require(type(fixed.get(k)) is int and fixed[k]==v,'Fixed experiment '+k)
    require(fixed.get('period')==['2016-08-17','2026-09-24'],'Fixed horizon')
    sources=p['sources']
    for key,path,digest in (('prices',PRICE,PRICE_SHA),('sessions',SESSIONS,SESSIONS_SHA),
                            ('calendar',CALENDAR,CALENDAR_SHA)):
        exact(sources[key],dict(path=path,sha256=digest))
        bound_ref(reader,sources[key],pins,key)
        if key in plan['inputs']:
            require(reader.path(plan['inputs'][key]['path'])==reader.path(path)
                and plan['inputs'][key]['sha256']==digest,'Original input '+key)
    for key in ('predecessor',):bound_ref(reader,sources[key],pins,key)
    return p


def axis(values):
    d=pd.DatetimeIndex(values)
    require(d.tz is None and not d.hasnans and d.is_unique and d.is_monotonic_increasing
        and d.equals(d.normalize()),'Ordered daily session axis')
    return d


def price_frames(payload,sessions,meta):
    require(set(payload)=={'dates','symbols','close','opening'},'Exact price schema')
    calendar,dates,signals=axis(sessions),axis(payload['dates']),axis(meta['dates'])
    require(len(calendar)>0 and len(dates)>0,'Nonempty calendar and prices')
    loc=calendar.get_indexer(signals)
    require((loc>=252).all(),'Full signal warmup')
    names=list(payload['symbols'])
    require(len(names)==len(set(names)) and all(isinstance(s,str) and s for s in names),'Named price columns')
    keep=(dates>=calendar[0]) & (dates<=calendar[-1]);inside=dates[keep]
    require(len(inside)>0 and inside.equals(calendar[(calendar>=inside[0])&(calendar<=inside[-1])]),'No missing official price rows')
    if len(signals):
        require(inside[0]<=calendar[int(min(loc))-252] and inside[-1]>=signals[-1],'Price coverage')
    require(all(set(pool)<=set(names) for pool in meta['symbols']),'Pool price columns')
    frames=[]
    for key in ('close','opening'):
        a=np.asarray(payload[key])
        require(a.shape==(len(dates),len(names)) and a.dtype==np.dtype('float64'),'Price shape and dtype')
        frames.append(pd.DataFrame(a[keep],index=inside,columns=names))
    return frames


def rank_oracle(values,available):
    """Pairwise counts, independent of producer sort/tie-block implementation."""
    values=np.asarray(values);mask=np.asarray(available)
    require(values.ndim==1 and mask.shape==values.shape and mask.dtype==bool,'Rank axes')
    valid=[float(x) for x in values[mask]]
    require(all(math.isfinite(x) for x in valid),'Finite observed rank values')
    out=np.zeros(values.shape)
    if len(valid)<2:return out
    for j in np.flatnonzero(mask):
        v=float(values[j]);less=sum(x<v for x in valid);same=sum(x==v for x in valid)
        out[j]=.5-(less+(same-1)/2)/(len(valid)-1)
    return out


def empty(reason):
    return dict(available=False,reason=reason,event_count=210,pair_count=209,
        event_ones=None,left_ones=None,right_ones=None,adjacent_one_pairs=None,
        covariance_numerator=None,frequency=None,cluster=None)


def event_oracle(close,opening):
    require(len(close)==211 and len(opening)==210,'Fixed price window')
    c=[float(x) for x in close];o=[float(x) for x in opening]
    if not all(math.isfinite(x) and x>0 for x in c+o):return empty('invalid_or_missing_window_price')
    events=[]
    for i in range(210):
        on,inside=o[i]/c[i],c[i+1]/o[i]
        if not math.isfinite(on) or on<=0 or not math.isfinite(inside) or inside<=0:
            return empty('unrepresentable_log_ratio')
        events.append(int(math.log(on)>1e-12 and math.log(inside)<-1e-12))
    # Sum products using Python integers; no NumPy count or covariance routine.
    total=sum(events);left=sum(events[:209]);right=sum(events[1:])
    joint=sum(events[i]*events[i+1] for i in range(209))
    numerator=209*joint-left*right
    return dict(available=True,reason='available',event_count=210,pair_count=209,
        event_ones=total,left_ones=left,right_ones=right,adjacent_one_pairs=joint,
        covariance_numerator=numerator,frequency=total/210,cluster=numerator/43681)


def features_oracle(meta,sessions,close,opening):
    calendar=axis(sessions);signals=axis(meta['dates'])
    require(len(signals)==len(meta['symbols']),'Aligned pools')
    for frame in (close,opening):
        require(isinstance(frame,pd.DataFrame) and frame.columns.is_unique,'Named price frame')
        require(axis(frame.index).isin(calendar).all(),'Prices on calendar')
    positions=calendar.get_indexer(signals);require((positions>=0).all(),'Signal sessions')
    arrays={k:np.full((len(signals),49),np.nan) for k in EXTRA[:2]}
    arrays.update({k:np.zeros((len(signals),49)) for k in HANDS})
    arrays['available']=np.zeros((len(signals),49),bool);cells=[];months=[]
    for i,(signal,pool,t) in enumerate(zip(signals,meta['symbols'],positions)):
        require(len(pool)==len(set(pool))==49 and all(type(s)is str and s for s in pool),'49 distinct names')
        enough=t>=252;window=calendar[t-252:t-41] if enough else calendar[:0]
        c=close.reindex(index=window,columns=pool)
        o=opening.reindex(index=window[1:],columns=pool)
        group=[]
        for j,symbol in enumerate(pool):
            stats=event_oracle(c[symbol],o[symbol]) if enough else empty('insufficient_history')
            row=dict(signal_index=i,pool_index=j,symbol=symbol,signal_date=str(signal.date()),**stats)
            group.append(row);arrays['available'][i,j]=stats['available']
            if stats['available']:
                for k in EXTRA[:2]:arrays[k][i,j]=stats[k]
        for raw,key in zip(EXTRA[:2],HANDS):arrays[key][i]=rank_oracle(arrays[raw][i],arrays['available'][i])
        for j,row in enumerate(group):
            row.update({k:float(arrays[k][i,j]) for k in HANDS})
        cells.extend(group)
        months.append(dict(signal_index=i,signal_date=str(signal.date()),
            close_window_start=str(window[0].date()) if enough else None,
            event_window_start=str(window[1].date()) if enough else None,
            window_end=str(window[-1].date()) if enough else None,
            available_candidates=int(arrays['available'][i].sum()),rank_available_candidates_only=True))
    return dict(arrays=arrays,cells=cells,months=months)


def validate_cells(cells,meta):
    require(type(cells)is list and len(cells)==49*len(meta['dates']),'Complete cells')
    groups=[];pos=0
    keys=set(empty('x'))|{'signal_index','pool_index','symbol','signal_date',*HANDS}
    for i,(day,pool) in enumerate(zip(meta['dates'],meta['symbols'])):
        group=[]
        for j,s in enumerate(pool):
            row=cells[pos];pos+=1
            require(type(row)is dict and set(row)==keys,'Exact event cell')
            require(type(row['signal_index'])is int and type(row['pool_index'])is int
                and (row['signal_index'],row['pool_index'],row['signal_date'],row['symbol'])==(i,j,day,s),'Cell identity')
            require(type(row['available'])is bool and row['event_count']==210 and row['pair_count']==209,'Cell counts')
            if row['available']:
                require(row['reason']=='available','Valid reason')
                for k in ('event_ones','left_ones','right_ones','adjacent_one_pairs','covariance_numerator'):
                    require(type(row[k])is int,'Exact integer event arithmetic')
                total,left,right,both=[row[k] for k in ('event_ones','left_ones','right_ones','adjacent_one_pairs')]
                require(0<=total<=210 and 0<=left<=209 and 0<=right<=209
                    and total-left in (0,1) and total-right in (0,1)
                    and max(0,left+right-209)<=both<=min(left,right),'Binary counts')
                numerator=209*both-left*right
                require(row['covariance_numerator']==numerator
                    and row['frequency']==total/210 and row['cluster']==numerator/43681,'Exact covariance and frequency')
            else:
                require(row['reason']in ('invalid_or_missing_window_price','unrepresentable_log_ratio','insufficient_history'),'Missing reason')
                require(all(row[k]is None for k in ('event_ones','left_ones','right_ones','adjacent_one_pairs','covariance_numerator','frequency','cluster')),'Missing raw values')
            group.append(row)
        mask=np.array([r['available'] for r in group],bool)
        for raw,key in zip(EXTRA[:2],HANDS):
            want=rank_oracle(np.array([r[raw] if r[raw]is not None else np.nan for r in group]),mask)
            require(all(type(r[key])is float for r in group),'Rank float types')
            np.testing.assert_array_equal([r[key] for r in group],want)
        groups.append(group)
    return groups


def feature_checks(feature,meta):
    for k,v in dict(method=METHOD,window_events=210,adjacent_pairs=209,close_rows=211,
        opening_rows=210,skip_sessions=42,width=49,signal_count=len(meta['dates']),sign_zero_tolerance=1e-12,
        rank_definition='0.5-average_tied_zero_based_rank/(n_valid-1); n_valid<=1 gives0',
        rank_direction='lower_frequency_or_covariance_is_higher',raw_missing='NaN',rank_missing=0,
        new_model_fits=0,covariance_arithmetic='integer_joint_count_times209_minus_left_times_right_over209_squared').items():
        exact(feature.get(k),v)
    for k in ('common_mask','constant_event_sequence_valid','covariance_uses_two209_pair_marginals',
              'caller_verified_complete_session_grid_required','caller_verified_common_adjusted_price_basis_required',
              'etfs_allowed','no_trading_eligibility_changes','dose_not_applied'):
        require(feature.get(k)is True,'Feature claim '+k)
    require(feature.get('portfolio_returns_or_nav_computed')is False,'No prepared returns')
    validate_cells(feature['cells'],meta)
    require(feature.get('reasons')==dict(Counter(c['reason'] for c in feature['cells'])),'Reason census')
    months=feature.get('months');require(type(months)is list and len(months)==len(meta['dates']),'All month windows')
    for i,row in enumerate(months):
        require(row['signal_index']==i and row['signal_date']==meta['dates'][i]
            and row['available_candidates']==sum(c['available'] for c in feature['cells'][i*49:(i+1)*49])
            and row['rank_available_candidates_only']is True,'Month rank census')
    exact(feature.get('feature_contract'),feature_contract())
    for key,sha in (('price_cache_sha256',PRICE_SHA),('sessions_sha256',SESSIONS_SHA),
                    ('calendar_plan_sha256',CALENDAR_SHA)):require(feature.get(key)==sha,'Feature provenance '+key)
