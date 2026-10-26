"""Independent fixed 21-session versus daily Sharpe and Local67 scoring oracle.

No new producer or host is imported. Only frozen independent pre-score and
accounting bridges are reused; source windows, statistics and ranking are new
scalar reconstructions.
"""
from datetime import date
import hashlib
import importlib.util
import math
import numbers
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'work/monthly_core_veto241_independent.py'
PRIOR_SHA='642958e877a0ed53347c2ca21aea0027e0148591af21985268976b91d9e22fb6'
PRIOR_TEST=ROOT/'work/test_monthly_core_veto241_independent.py'
PRIOR_TEST_SHA='4fc17fdcf385f30a87baf66513a1b2e0c19db2e0549dc779b6fe41f755e64087'
ARMS=('incumbent_local67','block21_sharpe','daily_sharpe_control')
VALUE_KEYS=('available','candidate','control','block_sharpe','daily_sharpe')
COUNT_FIELDS=('events','symbols_per_event','checked_cells','source_available_cells','available_cells',
    'unavailable_cells','valid_price_observations','daily_returns','block_returns','ranked_observations',
    'zero_daily_volatility_cells','zero_block_volatility_cells','new_model_fits','predictive_model_fits')
FEATURE_FLAGS=('passed','independent_scalar_rebuild','exact_official_253_price_window_checked',
    'skip42_checked','full_positive_source_checked','daily_252_returns_checked','disjoint_12_blocks_checked',
    'fixed_21_session_endpoints_checked','simple_return_ratios_checked','arithmetic_fsum_mean_checked',
    'two_pass_sample_std_checked','sample_ddof1_checked','common_positive_volatility_support_checked',
    'zero_volatility_common_exclusion_checked','common_full_pool_rank_domain_checked',
    'average_percentile_ties_checked','negative_means_valid','no_eligibility_filter',
    'full_audit_checked','descriptive_counts_checked','no_future_inputs','no_predictive_fits_checked')
DECISION_COUNT_FIELDS=('events','symbols_per_event','active_events','available_active_cells',
    'candidate_nonzero_delta_cells','control_nonzero_delta_cells','original_episode_calls','new_model_fits')
SELECTION_FLAGS=('passed','independent_original_pre_checked','original_model_rank_scope_preserved',
    'original_eligibility_unchanged','original_support_checked','original_budget_checked',
    'calm_nonweak_extension_budget_checked','half_core_formula_checked','core_weight_preserved',
    'missing_preserves_original_score','finite_zero_rank_active','calm_rows_unchanged',
    'named_initial_partial_unchanged','original_effects_unchanged','single_original_episode',
    'original_top_two_preserved','exact_scores_and_choices_checked','no_second_rank','full_audit_checked')


def prior():
    if hashlib.sha256(PRIOR.read_bytes()).hexdigest()!=PRIOR_SHA:
        raise ValueError('Frozen independent original-score/accounting bridge changed')
    spec=importlib.util.spec_from_file_location('block_sharpe242_independent_prior',PRIOR)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


old=prior()
require,exact_array,record_check=old.require,old.exact_array,old.record_check
PATHS,ACCOUNTING_SHA=old.PATHS,old.ACCOUNTING_SHA


def frozen_source_oracle():return old.frozen_source_oracle()
def accounting():return old.accounting()


def dependency_files():
    require(hashlib.sha256(PRIOR_TEST.read_bytes()).hexdigest()==PRIOR_TEST_SHA,'Frozen synthetic fixture changed')
    return tuple(sorted(set(old.dependency_files())|{PRIOR,PRIOR_TEST,Path(__file__).resolve()}))


def finite(value,label='derived statistic'):
    if not math.isfinite(value):raise FloatingPointError('Nonfinite '+label)
    return float(value)


def total(values):
    try:return finite(math.fsum(values),'sum')
    except (OverflowError,ValueError) as exc:raise FloatingPointError('Invalid finite-input sum') from exc


def iso(raw):
    require(type(raw) is str,'Canonical ISO date required')
    d=date.fromisoformat(raw);require(d.isoformat()==raw,'Canonical ISO date required')
    return pd.Timestamp(d)


def axes(panel,meta,calendar,limit=None):
    require(isinstance(panel,dict) and set(panel)=={'adj'},'Only original adjusted price frame')
    frame=panel['adj'];require(isinstance(frame,pd.DataFrame) and isinstance(frame.index,pd.DatetimeIndex),'Price frame')
    ix=frame.index
    require(not ix.hasnans and ix.is_unique and ix.is_monotonic_increasing and ix.tz is None and
            ix.equals(ix.normalize()),'Unique ordered naive price sessions')
    require(frame.columns.is_unique and all(type(s) is str and s and s.strip()==s for s in frame.columns),'Price IDs')
    require(isinstance(meta,dict) and isinstance(meta.get('dates'),list) and isinstance(meta.get('symbols'),list)
            and len(meta['dates'])==len(meta['symbols']),'Full query axes')
    full=len(meta['dates']);n=full if limit is None else limit
    require(type(n) is int and 0<=n<=full,'Consumed prefix')
    etfs=meta.get('etfs')
    require(isinstance(etfs,(list,tuple,set,frozenset)) and len(etfs)==len(set(etfs)) and
            all(type(s) is str and s and s.strip()==s for s in etfs),'ETF metadata')
    require(isinstance(calendar,(list,tuple)) and (calendar or n==0),'Official calendar')
    sessions=[iso(d) for d in calendar]
    require(all(d.weekday()<5 for d in sessions) and all(a<b for a,b in zip(sessions,sessions[1:])),
            'Official session order')
    lookup={d:i for i,d in enumerate(sessions)};queries=[]
    width=len(meta['symbols'][0]) if full else 0
    for i,(raw,names) in enumerate(zip(meta['dates'][:n],meta['symbols'][:n])):
        d=iso(raw);require(d in lookup and (not queries or d>queries[-1][0]),'Ordered official query dates')
        require(isinstance(names,(list,tuple)) and len(names)==width and width>0 and len(set(names))==width
                and all(type(s) is str and s and s.strip()==s for s in names),'Unique original query IDs')
        t=lookup[d]
        if raw=='2016-08-17':require(i==0,'Named partial only first')
        else:require(t+1<len(sessions) and sessions[t+1].to_period('M')!=d.to_period('M'),'Official month-end witness')
        queries.append((d,list(names),t))
    return frame,sessions,queries,width


def percentile(raw,mask):
    out=np.full(mask.shape,np.nan)
    for i in range(len(mask)):
        ids=[j for j in range(mask.shape[1]) if mask[i,j]];n=len(ids)
        for j in ids:
            less=sum(raw[i,k]<raw[i,j] for k in ids)
            equal=sum(raw[i,k]==raw[i,j] for k in ids)
            out[i,j]=(less+(equal+1)/2.)/n-.5
    return out


STAT_FIELDS=('start_price','end_price','daily_return_sum','daily_mean','daily_centered_square_sum',
    'daily_sample_vol','block_return_sum','block_mean','block_centered_square_sum','block_sample_vol',
    'daily_sharpe','block_sharpe')


def empty(reason):
    return dict(source_available=False,available=False,reason=reason,**{k:None for k in STAT_FIELDS})


def sample_statistics(returns):
    count=len(returns);require(count>=2,'At least two returns')
    summed=total(returns);mean=finite(summed/float(count))
    if summed!=0. and mean==0.:raise FloatingPointError('Nonzero mean underflow')
    squared=[]
    for r in returns:
        deviation=finite(r-mean);square=finite(deviation*deviation)
        if deviation!=0. and square==0.:raise FloatingPointError('Centered square underflow')
        squared.append(square)
    centered=total(squared);variance=finite(centered/float(count-1))
    if centered>0. and variance==0.:raise FloatingPointError('Sample variance underflow')
    return summed,mean,centered,finite(math.sqrt(variance))


def simple_return(end,start):
    ratio=finite(end/start,'positive price ratio')
    if ratio==0.:raise FloatingPointError('Positive price ratio underflow')
    return finite(ratio-1.,'simple return')


def normalized_mean(mean,sd):
    ratio=finite(mean/sd,'Sharpe ratio')
    if mean!=0. and ratio==0.:raise FloatingPointError('Nonzero Sharpe underflow')
    return ratio


def measure_oracle(prices):
    require(np.shape(prices)==(253,),'Exactly 253 stored prices')
    if any(not isinstance(v,numbers.Real) or isinstance(v,(bool,np.bool_)) for v in prices):
        return empty('nonnumeric_adjusted_price')
    try:p=[float(v) for v in prices]
    except (TypeError,ValueError,OverflowError):return empty('nonnumeric_adjusted_price')
    if any(not math.isfinite(v) or v<=0. for v in p):return empty('invalid_adjusted_price')
    try:
        daily=[simple_return(p[i],p[i-1]) for i in range(1,253)]
        blocks=[simple_return(p[i+21],p[i]) for i in range(0,252,21)]
        dr,dm,ds,dv=sample_statistics(daily);br,bm,bs,bv=sample_statistics(blocks)
        good=dv>0. and bv>0.
        reason=('available' if good else 'zero_both_sample_volatility' if dv==0. and bv==0.
                else 'zero_daily_sample_volatility' if dv==0. else 'zero_block_sample_volatility')
        out=empty(reason);out.update(source_available=True,available=good,start_price=p[0],end_price=p[-1],
            daily_return_sum=dr,daily_mean=dm,daily_centered_square_sum=ds,daily_sample_vol=dv,
            block_return_sum=br,block_mean=bm,block_centered_square_sum=bs,block_sample_vol=bv)
        if good:out.update(daily_sharpe=normalized_mean(dm,dv),block_sharpe=normalized_mean(bm,bv))
        return out
    except (OverflowError,ValueError,ZeroDivisionError) as exc:
        raise FloatingPointError('Invalid finite positive source arithmetic') from exc


def independent_features(panel,meta,calendar,limit=None):
    frame,sessions,queries,width=axes(panel,meta,calendar,limit)
    shape=(len(queries),width);values={k:np.full(shape,np.nan) for k in VALUE_KEYS[1:]}
    values['available']=np.zeros(shape,bool);history=[];reasons={}
    counts={k:0 for k in COUNT_FIELDS};counts.update(events=len(queries),symbols_per_event=width,
                                                   checked_cells=len(queries)*width)
    for i,(day,names,t) in enumerate(queries):
        dates=sessions[t-294:t-41] if t>=294 else []
        missing=[str(d.date()) for d in dates if d not in frame.index];cells=[]
        for j,symbol in enumerate(names):
            reason=('insufficient_history' if not dates else 'missing_symbol' if symbol not in frame.columns
                    else 'missing_official_session' if missing else None)
            measured=empty(reason) if reason else measure_oracle([frame.at[d,symbol] for d in dates])
            cell=dict(column_index=j,symbol=symbol,**measured,candidate=None,control=None)
            source=measured['source_available'];good=measured['available']
            values['available'][i,j]=good
            if source:
                counts['source_available_cells']+=1
                counts['zero_daily_volatility_cells']+=int(measured['daily_sample_vol']==0.)
                counts['zero_block_volatility_cells']+=int(measured['block_sample_vol']==0.)
            if good:
                counts['available_cells']+=1
                for k in ('block_sharpe','daily_sharpe'):values[k][i,j]=measured[k]
            reasons[cell['reason']]=reasons.get(cell['reason'],0)+1;cells.append(cell)
        history.append(dict(query_index=i,date=str(day.date()),symbols=names,session_position=t,
            price_start=str(dates[0].date()) if dates else None,price_end=str(dates[-1].date()) if dates else None,
            source_dates=[str(d.date()) for d in dates],block_endpoint_dates=[str(d.date()) for d in dates[::21]],
            missing_official_sessions=missing,source_available_indices=[j for j,c in enumerate(cells) if c['source_available']],
            available_indices=[j for j,c in enumerate(cells) if c['available']],cells=cells))
    values['candidate']=percentile(values['block_sharpe'],values['available'])
    values['control']=percentile(values['daily_sharpe'],values['available'])
    for i,event in enumerate(history):
        for j,cell in enumerate(event['cells']):
            if cell['available']:cell.update(candidate=float(values['candidate'][i,j]),control=float(values['control'][i,j]))
    source=counts['source_available_cells'];available=counts['available_cells']
    counts.update(unavailable_cells=counts['checked_cells']-available,valid_price_observations=253*source,
        daily_returns=252*source,block_returns=12*source,ranked_observations=2*available)
    return values,dict(schema='monthly_block_sharpe242_features_v1',counts=counts,reason_counts=reasons,event_history=history)


def verify_features(panel,meta,calendar,values,audit,limit=None):
    expected,report=independent_features(panel,meta,calendar,limit)
    require(isinstance(values,dict) and set(values)==set(VALUE_KEYS),'Exactly five features')
    for k in VALUE_KEYS:exact_array(values[k],expected[k],'Independent '+k)
    record_check(audit,report,'Independent entire feature audit')
    return dict(**{k:True for k in FEATURE_FLAGS},**report['counts'],counts=report['counts'],
        numerical_tolerance=0.,producer_modules_imported=False,computes_returns=False)


def selection_values(values,n,width):
    require(isinstance(values,dict) and set(values)==set(VALUE_KEYS),'Five feature arrays')
    mask=np.asarray(values['available']);require(mask.ndim==2 and mask.dtype==np.dtype(bool) and
        len(mask)>=n and (mask.shape[1]==width or n==0 and mask.shape==(0,0)),'Feature availability axis')
    if n==0:return {'available':np.empty((0,width),bool),**{k:np.empty((0,width)) for k in VALUE_KEYS[1:]}}
    out={'available':mask[:n].copy()}
    for k in VALUE_KEYS[1:]:
        v=np.asarray(values[k]);require(v.shape==mask.shape and v.dtype==np.dtype(np.float64),'Feature float64 axis '+k)
        v=v[:n];require(np.isfinite(v[out['available']]).all() and np.isnan(v[~out['available']]).all(),
                         'Common finite feature support '+k);out[k]=v.copy()
    exact_array(out['candidate'],percentile(out['block_sharpe'],out['available']),'Block percentile rank')
    exact_array(out['control'],percentile(out['daily_sharpe'],out['available']),'Daily percentile rank')
    return out


def _selection(original,meta,values,n=None):
    n=len(meta['dates']) if n is None else n
    require(type(n) is int and 0<=n<=len(meta['dates']),'Selection prefix')
    width,days,names=old.axes(original,meta,n);v=selection_values(values,n,width)
    dc={k:0 for k in DECISION_COUNT_FIELDS};dc.update(events=n,symbols_per_event=width,original_episode_calls=3)
    decision=dict(schema='monthly_block_sharpe242_decisions_v1',counts=dc,events=[])
    if n==0:
        empty={arm:np.empty((0,width)) for arm in ARMS}
        return empty,{arm:[] for arm in ARMS},{arm:a.copy() for arm,a in empty.items()},decision,dict(
            minimum_remaining_support=None,maximum_budget_sum_error=0.,calm_nonweak_extension_count=0,
            maximum_absolute_delta=0.,changed_cells={k:0 for k in ARMS})
    x=frozen_source_oracle().old.original_helper().original_prefix(original,meta,n)
    exact_array(x['cw'],np.broadcast_to(np.where(x['calm'],.5,.52)[:,None],(n,width)), 'Original core weight')
    require((np.abs(x['core'])<=.5*x['cw']).all(),'Original weighted core bounds')
    ext=np.asarray(original['extension_available'])[:n];forecast=np.asarray(original['forecast_available'])[:n]
    ranklog=np.asarray(original['rank_log_excess'])[:n]
    require(ext.shape==forecast.shape==ranklog.shape==(n,width) and ext.dtype==forecast.dtype==np.dtype(bool),
            'Original extension masks')
    A=(x['calm']&~x['weak'])[:,None]&ext&forecast&x['eligible']&np.isfinite(ranklog)
    budget=np.minimum(.05,np.maximum(0.,.10-(.025*x['calm']+.05*x['weak'])))
    active=~x['calm']
    for i,d in enumerate(days):
        if str(d.date())=='2016-08-17':active[i]=False
    scores={};choices={};pres={};changed={};all_deltas={};maxdelta=0.;minimum=1.;sum_error=0.
    for arm,key in zip(ARMS,(None,'candidate','control')):
        delta=np.zeros((n,width));pre=x['local'].copy()
        if key is not None:
            for i in range(n):
                if active[i]:
                    for j in range(width):
                        if v['available'][i,j]:
                            delta[i,j]=.5*(float(x['cw'][i,j])*float(v[key][i,j])-float(x['core'][i,j]))
        require(np.isfinite(delta).all() and (np.abs(delta)<=.26).all(),'Finite half-core delta')
        nonzero=delta!=0.;pre[nonzero]+=delta[nonzero]
        require(np.isfinite(pre).all(),'Finite modified pre')
        changed[arm]=int(nonzero.sum());maxdelta=max(maxdelta,float(np.max(np.abs(delta))))
        all_deltas[arm]=delta
        pres[arm]=pre.copy();score=pre.copy();rows=[];bound=np.zeros((n,width),bool)
        for i,ss in enumerate(names):
            order=sorted([j for j in range(width) if x['eligible'][i,j]],key=lambda j:(-pre[i,j],ss[j]))
            if len(order)<3:rows.append(order if len(order)==2 else []);continue
            J=[j for j in order[2:5] if float(pre[i,order[2]])-float(pre[i,j])<=.02]
            for j in J:
                bound[i,j]=True
                target=min(.5,max(-.5,x['support'][i,j]+x['effects'][i,j]))
                score[i,j]+=budget[i]*(target-x['support'][i,j])
            winner=min(J,key=lambda j:(-score[i,j],ss[j]))
            rows.append(sorted(order[:2]+[winner],key=lambda j:(-score[i,j],ss[j])))
        require(np.isfinite(score).all(),'Finite original episode scores')
        for i in range(n):
            for j in range(width):
                weights=[float(x['cw'][i,j]),.025*float(x['calm'][i]),.05*float(x['weak'][i]),
                    .01*float(A[i,j]),float(budget[i])*float(bound[i,j])]
                remaining=1.-weights[0]
                for w in weights[1:]:remaining-=w
                error=abs(math.fsum(weights+[remaining])-1.)
                require(remaining>=.38-1e-12 and error<=1e-12,'Unchanged Local67 support budget')
                minimum=min(minimum,remaining);sum_error=max(sum_error,error)
        scores[arm]=score;choices[arm]=rows
    dc.update(active_events=int(active.sum()),available_active_cells=int((v['available']&active[:,None]).sum()),
        candidate_nonzero_delta_cells=changed[ARMS[1]],control_nonzero_delta_cells=changed[ARMS[2]])
    for i,d in enumerate(days):
        decision['events'].append(dict(query_index=i,date=meta['dates'][i],initial_partial=str(d.date())=='2016-08-17',
            calm=bool(x['calm'][i]),active=bool(active[i]),available_indices=np.flatnonzero(v['available'][i]).tolist(),
            candidate_delta=all_deltas[ARMS[1]][i].tolist(),control_delta=all_deltas[ARMS[2]][i].tolist()))
    return scores,choices,pres,decision,dict(minimum_remaining_support=minimum,maximum_budget_sum_error=sum_error,
        calm_nonweak_extension_count=int(A.sum()),maximum_absolute_delta=maxdelta,changed_cells=changed)


def selection_oracle(original,meta,values,n=None):return _selection(original,meta,values,n)[:4]


def verify_selection(original,meta,values,orders,*,scores,pre_scores,audit,n=None):
    n=len(meta['dates']) if n is None else n
    expected,choices,pres,expected_audit,stats=_selection(original,meta,values,n)
    wanted={arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
        symbols=[meta['symbols'][i][j] for j in ids],scores=[float(expected[arm][i,j]) for j in ids])
        for i,ids in enumerate(choices[arm])] for arm in ARMS}
    record_check(orders,wanted,'Independent ordered selection')
    for got,ref,label in ((scores,expected,'full scores'),(pre_scores,pres,'full pre-scores')):
        require(isinstance(got,dict) and set(got)==set(ARMS),'Three '+label)
        for arm in ARMS:exact_array(got[arm],ref[arm],label+'/'+arm)
    record_check(audit,expected_audit,'Independent decision audit')
    return dict(**{k:True for k in SELECTION_FLAGS},**stats,counts=expected_audit['counts'],checked_events=n,ordered_choices_checked=3*n,
        full_scores_supplied=True,full_pre_scores_supplied=True,full_audit_supplied=True,numerical_tolerance=0.,
        original_source_parity_tolerance=1e-12,producer_modules_imported=False,computes_returns=False)


def audit_outcomes(px,opening,orders,exports,results,meta):
    require(set(orders)==set(exports)==set(results)==set(ARMS),'Outcome arms')
    mapping=dict(zip(old.ARMS,ARMS))
    report=prior().audit_outcomes(px,opening,*[{k:data[v] for k,v in mapping.items()}
        for data in (orders,exports,results)],meta)
    report['paths']={mapping[k]:v for k,v in report['paths'].items()};return report
