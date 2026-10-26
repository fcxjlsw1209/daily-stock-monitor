"""Independent source gate, unchanged Local67 pre-score, and two-arm accounts.

New arithmetic is reconstructed without importing the production gate/host.
Only pinned independent original-score and accounting primitives are reused.
No file output, model training, or real-data entry point is provided.
"""
from datetime import date
import hashlib
import importlib.util
import math
import numbers
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HELPER = ROOT / 'work/monthly_endpoint155_prepare_review.py'
SOURCE_SHA = '0e594355beecc84a705517c8cd2774214e4300b5248e0c772a0306e73c38174f'
ACCOUNTING = ROOT / 'work/ptr169_results_independent.py'
ACCOUNTING_SHA = '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59'
ARMS = ('incumbent_local67', 'core_horizon_gate')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
VALUE_KEYS = ('original_eligible','candidate_eligible','required','warm','momentum210','momentum252')
COUNT_FIELDS = ('events','symbols_per_event','checked_cells','original_eligible_cells',
    'calm_events','active_events','required_cells','warm_cells','warm_observations',
    'source_price_observations','added_cells','rejected_cells','candidate_eligible_cells',
    'new_model_fits','predictive_model_fits')
FEATURE_FLAGS = ('passed','independent_scalar_rebuild','original_eligibility_exact',
    'original_calm_state_preserved','initial_partial_unchanged','official_endpoint_offsets_checked',
    'official_panel295_axis_checked','warm295_notna_semantics_checked','current_positive_gate_checked',
    'positive_nonboolean_endpoints_checked','required_source_only_checked','missing_required_source_fails',
    'original210_eligibility_parity_checked','core252_sign_checked','complete_replacement_checked',
    'no_eligibility_rerank','no_model_or_score_changes','audit_checked','descriptive_counts_checked',
    'no_predictive_fits_checked','no_future_or_outcome_inputs')
SELECTION_FLAGS = ('passed','independent_original_pre_checked','original_pre_unchanged',
    'original_model_rank_scope_preserved','original_support_checked','original_budget_checked',
    'original_effects_unchanged','final_eligibility_only_checked','complete_replacement_checked',
    'one_episode_per_arm','initial_partial_unchanged','exact_scores_and_choices_checked')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pinned(path, digest, name):
    require(sha(path) == digest, 'Frozen independent dependency changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def frozen_source_oracle():
    return load_pinned(SOURCE_HELPER, SOURCE_SHA, 'core_gate236_original_score')


def accounting():
    return load_pinned(ACCOUNTING, ACCOUNTING_SHA, 'core_gate236_accounts')


def dependency_files():
    return tuple(sorted(set(frozen_source_oracle().source_files(ROOT)) |
                        {Path(__file__).resolve(), SOURCE_HELPER, ACCOUNTING,
                         ROOT/'work/test_monthly_upside_retention181_independent.py',
                         ROOT/'work/monthly_upside_retention181_independent.py'}))


def exact_array(actual, expected, label):
    a,e=np.asarray(actual),np.asarray(expected)
    require(a.shape==e.shape and a.dtype==e.dtype and a.tobytes()==e.tobytes(), label+': exact array bytes')


def record_check(actual, expected, label='audit'):
    if isinstance(expected, dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),label+': keys')
        for key in expected: record_check(actual[key],expected[key],label+'/'+key)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),label+': list')
        for i,(a,b) in enumerate(zip(actual,expected)):record_check(a,b,label+'/'+str(i))
    elif isinstance(expected,float):
        require(type(actual) is float and np.float64(actual).tobytes()==np.float64(expected).tobytes(),label+': float bytes')
    else:
        require(type(actual) is type(expected) and actual==expected,label+': value/type')


def iso(raw):
    require(isinstance(raw,str),'Canonical ISO date required')
    d=date.fromisoformat(raw)
    require(d.isoformat()==raw,'Canonical ISO date required')
    return pd.Timestamp(d)


def axes(arrays, meta, limit=None):
    require(isinstance(arrays,dict) and all(k in arrays for k in ('eligible','state_calm10')),
            'Original eligibility and calm-state arrays required')
    e,w=np.asarray(arrays['eligible']),np.asarray(arrays['state_calm10'])
    require(e.ndim==2 and e.shape[1]>0,'Eligibility requires nonempty candidate width')
    full,width=e.shape
    require(w.shape==(full,),'Calm-state shape must be [months]')
    require(limit is None or type(limit) is int and 0<=limit<=full,'Invalid physical prefix')
    n=full if limit is None else limit
    require(all(isinstance(v,(bool,np.bool_)) for v in e[:n].flat) and
            all(isinstance(v,(bool,np.bool_)) for v in w[:n].flat),'Exact consumed boolean source values required')
    require(isinstance(meta,dict) and isinstance(meta.get('dates'),list) and isinstance(meta.get('symbols'),list)
            and len(meta['dates'])==len(meta['symbols'])==full,'Metadata axes differ')
    names=[];days=[]
    for raw,ss in zip(meta['dates'][:n],meta['symbols'][:n]):
        day=iso(raw)
        require(not days or day>days[-1],'Query dates must increase')
        require(isinstance(ss,list) and len(ss)==width and len(set(ss))==width and
                all(isinstance(s,str) and s and s.strip()==s for s in ss),'Unique query identities required')
        days.append(day);names.append(ss.copy())
    return e[:n].astype(bool,copy=True),w[:n].astype(bool,copy=True),days,names,n,width


def source_axes(panel, calendar, days):
    require(isinstance(panel,dict) and set(panel)=={'adj'},'Exactly adjusted-price panel required')
    frame=panel['adj']
    require(isinstance(frame,pd.DataFrame) and isinstance(frame.index,pd.DatetimeIndex),'Daily adjusted-price frame required')
    idx=frame.index
    require(not idx.hasnans and idx.is_unique and idx.is_monotonic_increasing and idx.tz is None and
            idx.equals(idx.normalize()),'Invalid price date axis')
    require(frame.columns.is_unique and all(isinstance(s,str) and s and s.strip()==s for s in frame.columns),
            'Invalid price identity axis')
    require(isinstance(calendar,(list,tuple)) and (len(calendar)>0 or not days),'Nonempty official calendar required for queries')
    sessions=[iso(d) for d in calendar]
    require(all(d.weekday()<5 for d in sessions) and all(a<b for a,b in zip(sessions,sessions[1:])),
            'Invalid official calendar order')
    lookup={d:i for i,d in enumerate(sessions)};positions=[]
    for i,day in enumerate(days):
        require(day in lookup,'Query absent from official calendar')
        t=lookup[day];partial=day==pd.Timestamp('2016-08-17')
        if partial:require(i==0,'Initial partial may only be first query')
        else:require(t+1<len(sessions) and sessions[t+1].to_period('M')!=day.to_period('M'),
                     'Official month-end requires next-session witness')
        positions.append(t)
    return frame,sessions,positions


def price(frame, day, symbol):
    require(symbol in frame.columns and day in frame.index,'Required source endpoint missing: '+symbol+'/'+str(day.date()))
    raw=frame.at[day,symbol]
    require(isinstance(raw,numbers.Real) and not isinstance(raw,(bool,np.bool_)),'Nonboolean real endpoint required')
    try:value=float(raw)
    except (ValueError,TypeError,OverflowError) as exc:raise ValueError('Invalid required endpoint') from exc
    require(math.isfinite(value) and value>0.,'Positive finite endpoint required')
    return value


def momentum(start,end):
    for value in (start,end):
        require(isinstance(value,numbers.Real) and not isinstance(value,(bool,np.bool_)), 'Real endpoint required')
        try:converted=float(value)
        except (TypeError,ValueError,OverflowError) as exc:raise ValueError('Invalid endpoint') from exc
        require(math.isfinite(converted) and converted>0.,'Positive finite endpoint required')
    value=float(end)/float(start)-1.
    if not math.isfinite(value):raise FloatingPointError('Nonfinite endpoint momentum')
    return value


def independent_features(panel,meta,calendar,arrays,limit=None):
    old,calm,days,names,n,width=axes(arrays,meta,limit)
    frame,sessions,positions=source_axes(panel,calendar,days)
    values={key:np.zeros((n,width),bool) for key in VALUE_KEYS[:4]}
    values['original_eligible']=old.copy();values['candidate_eligible']=old.copy()
    for key in VALUE_KEYS[4:]:values[key]=np.full((n,width),np.nan)
    counts={key:0 for key in COUNT_FIELDS};counts.update(events=n,symbols_per_event=width,checked_cells=n*width)
    events=[]
    for i,(day,ss,t) in enumerate(zip(days,names,positions)):
        partial=day==pd.Timestamp('2016-08-17');active=bool(not calm[i] and not partial)
        ids=[j for j in range(width) if old[i,j]];required=list(range(width)) if active else []
        source_positions=[];source_dates=[];warm_ids=[]
        if active:
            require(t>=294,'Insufficient official warm history')
            expected_axis=pd.DatetimeIndex(sessions[t-294:t+1])
            source=frame.loc[expected_axis[0]:day]
            require(source.index.equals(expected_axis),'Panel warm295 axis must exactly match official sessions')
            require(all(name in frame.columns for name in ss),'Required security column missing')
            source_positions=[t-294,t-252,t-42,t]
            source_dates=[str(sessions[k].date()) for k in source_positions]
            values['required'][i,:]=True
            for j,name in enumerate(ss):
                path=source[name].tolist()
                # This is the original notna warm rule, not an all-positive/finite filter.
                if any(bool(pd.isna(x)) for x in path):
                    warm=False
                else:
                    current=path[-1]
                    require(isinstance(current,numbers.Real) and not isinstance(current,(bool,np.bool_)),
                            'Nonboolean real current price required')
                    try:current=float(current)
                    except (ValueError,TypeError,OverflowError) as exc:raise ValueError('Invalid current price') from exc
                    require(math.isfinite(current),'Finite current price required')
                    warm=bool(current>0.)
                values['warm'][i,j]=warm
                if warm:
                    endpoints=[price(frame,sessions[k],name) for k in source_positions]
                    m210=momentum(endpoints[1],endpoints[2]);m252=momentum(endpoints[0],endpoints[2])
                    values['momentum210'][i,j]=m210;values['momentum252'][i,j]=m252
                    require(bool(old[i,j])==bool(m210>0.),'Original E disagrees with warm and M210 sign')
                    values['candidate_eligible'][i,j]=m252>0.;warm_ids.append(j)
                else:
                    require(not old[i,j],'Original E must be false when warm295 is false')
                    values['candidate_eligible'][i,j]=False
        chosen=[j for j in range(width) if values['candidate_eligible'][i,j]]
        added=[j for j in chosen if not old[i,j]];removed=[j for j in ids if not values['candidate_eligible'][i,j]]
        events.append(dict(query_index=i,date=str(day.date()),symbols=ss,session_position=t,
            calm10=bool(calm[i]),initial_partial=bool(partial),active=active,
            branch='initial_partial' if partial else ('core_horizon_gate' if active else 'calm_bypass'),
            source_positions=source_positions,source_dates=source_dates,
            original_eligible_indices=ids,candidate_eligible_indices=chosen,required_indices=required,
            warm_indices=warm_ids,added_indices=added,rejected_indices=removed))
        counts['original_eligible_cells']+=len(ids);counts['calm_events']+=int(calm[i])
        counts['active_events']+=int(active);counts['required_cells']+=len(required)
        counts['warm_cells']+=len(warm_ids);counts['warm_observations']+=295*len(required)
        counts['source_price_observations']+=4*len(warm_ids);counts['added_cells']+=len(added)
        counts['rejected_cells']+=len(removed);counts['candidate_eligible_cells']+=len(chosen)
    audit=dict(schema='monthly_core_gate236_features_v1',**counts,warm_price_count=295,
        old_start_offset=252,new_start_offset=294,end_offset=42,old_return_count=210,new_return_count=252,
        rankings_recomputed=False,source_missing_filled=False,portfolio_returns_calculated=False,event_history=events)
    return values,audit


def values_check(values,expected):
    require(isinstance(values,dict) and set(values)==set(VALUE_KEYS),'Exactly six gate evidence arrays required')
    for key in VALUE_KEYS:exact_array(values[key],expected[key],'Independent '+key)


def verify_features(panel,meta,calendar,arrays,values,audit,limit=None):
    expected,ea=independent_features(panel,meta,calendar,arrays,limit)
    values_check(values,expected);record_check(audit,ea)
    counts={key:ea[key] for key in COUNT_FIELDS}
    return dict(**{key:True for key in FEATURE_FLAGS},**counts,counts=counts,numerical_tolerance=0.,
                producer_modules_imported=False,computes_returns=False)


def selection_evidence(original,meta,values,n):
    old,calm,days,_,_,width=axes(original,meta,n)
    require(isinstance(values,dict) and set(values)==set(VALUE_KEYS),'Gate evidence keys')
    v={k:np.asarray(values[k])[:n] for k in VALUE_KEYS}
    for key in VALUE_KEYS:
        dtype=np.dtype(bool) if key in VALUE_KEYS[:4] else np.dtype(np.float64)
        require(v[key].shape==(n,width) and v[key].dtype==dtype,'Gate evidence dtype/shape: '+key)
    exact_array(v['original_eligible'],old,'Original E selection identity')
    want=old.copy();required=np.zeros_like(old);activewarm=np.zeros_like(old)
    for i,day in enumerate(days):
        active=bool(not calm[i] and day!=pd.Timestamp('2016-08-17'))
        if active:
            required[i,:]=True;activewarm[i]=v['warm'][i]
            for j in range(width):
                warm=bool(v['warm'][i,j])
                if warm:
                    m210=float(v['momentum210'][i,j]);m252=float(v['momentum252'][i,j])
                    require(math.isfinite(m210) and math.isfinite(m252),'Required finite momentum evidence')
                    require(bool(old[i,j])==bool(m210>0.),'Original E warm/M210 source parity')
                    want[i,j]=m252>0.
                else:
                    require(not old[i,j],'Original E nonwarm source parity');want[i,j]=False
        else:require(not v['warm'][i].any(),'Bypassed query warm evidence must remain false')
    exact_array(v['required'],required,'Required warm reconstruction domain')
    for key in VALUE_KEYS[4:]:require(np.isnan(v[key][~activewarm]).all(),'Unused momentum must remain NaN')
    exact_array(v['candidate_eligible'],want,'Canonical complete core-horizon replacement')
    return v,old,width


def selection_oracle(original,meta,values,n=None):
    n=len(meta['dates']) if n is None else n
    require(type(n) is int and 0<=n<=len(meta['dates']),'Selection prefix')
    v,old,width=selection_evidence(original,meta,values,n)
    if n==0:
        return ({arm:np.empty((0,width)) for arm in ARMS},{arm:[] for arm in ARMS},
                {arm:np.empty((0,width)) for arm in ARMS})
    x=frozen_source_oracle().old.original_helper().original_prefix(original,meta,n)
    exact_array(x['eligible'],old,'Frozen original model eligibility')
    require(x['local'].shape==(n,width),'Original pre shape')
    budget=np.minimum(.05,np.maximum(0.,.10-(.025*x['calm']+.05*x['weak'])))
    scores={};choices={};pres={}
    for arm in ARMS:
        pre=x['local'].copy();pres[arm]=pre.copy();score=pre.copy();rows=[]
        gate=old if arm==ARMS[0] else v['candidate_eligible']
        for i,names in enumerate(x['names']):
            order=sorted([j for j in range(width) if gate[i,j]],key=lambda j:(-pre[i,j],names[j]))
            if len(order)<3:rows.append(order if len(order)==2 else []);continue
            boundary=[j for j in order[2:5] if float(pre[i,order[2]])-float(pre[i,j])<=.02]
            for j in boundary:
                target=min(.5,max(-.5,x['support'][i,j]+x['effects'][i,j]))
                score[i,j]+=budget[i]*(target-x['support'][i,j])
            winner=min(boundary,key=lambda j:(-score[i,j],names[j]))
            rows.append(sorted(order[:2]+[winner],key=lambda j:(-score[i,j],names[j])))
        scores[arm]=score;choices[arm]=rows
    return scores,choices,pres


def verify_selection(original,meta,values,orders,*,scores=None,pre_scores=None,n=None):
    n=len(meta['dates']) if n is None else n
    expected,choices,pres=selection_oracle(original,meta,values,n)
    require(set(orders)==set(ARMS),'Two order arms required')
    wanted={arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
        symbols=[meta['symbols'][i][j] for j in ids],scores=[float(expected[arm][i,j]) for j in ids])
        for i,ids in enumerate(choices[arm])] for arm in ARMS}
    record_check(orders,wanted,'Independent ordered selection')
    for supplied,reference,label in ((scores,expected,'post-episode score'),(pre_scores,pres,'original pre-score')):
        if supplied is not None:
            require(isinstance(supplied,dict) and set(supplied)==set(ARMS),'Two '+label+' arms')
            for arm in ARMS:exact_array(supplied[arm],reference[arm],label+'/'+arm)
    return dict(**{key:True for key in SELECTION_FLAGS},ordered_choices_checked=2*n,checked_events=n,
        full_scores_supplied=scores is not None,full_pre_scores_supplied=pre_scores is not None,
        numerical_tolerance=0.,original_source_parity_tolerance=1e-12,
        producer_modules_imported=False,computes_returns=False,
        holdings_counts={arm:{str(k):sum(len(row)==k for row in choices[arm]) for k in (0,2,3)} for arm in ARMS})


def audit_outcomes(px,opening,orders,exports,results,meta):
    """Two arms, three execution paths and five saved slices; no engine reuse."""
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
    return dict(passed=True,checked_paths=6,checked_slices=30,independent_accounts=True,
        independent_of_production_model_and_engine=True,terminal_liquidation=False,
        accounting_source_sha256=ACCOUNTING_SHA,paths=reports)
