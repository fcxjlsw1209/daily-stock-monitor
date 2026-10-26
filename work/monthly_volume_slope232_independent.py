"""Independent stored-volume trend and unchanged-episode verification.

No producer feature, ranking, selection or accounting implementation is imported.
The frozen independent source helper reconstructs the original Local67 arrays;
new arithmetic and episode application are rebuilt here from supplied inputs.
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
SOURCE_HELPER = ROOT/'work/monthly_endpoint155_prepare_review.py'
SOURCE_SHA = '0e594355beecc84a705517c8cd2774214e4300b5248e0c772a0306e73c38174f'
ACCOUNTING = ROOT/'work/ptr169_results_independent.py'
ACCOUNTING_SHA = '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59'
ARMS = ('incumbent_local67', 'volume_rank_slope', 'unweighted_slope_control')
VALUE_KEYS = ('candidate', 'control', 'available')
ARRAY_KEYS = VALUE_KEYS
COUNT_FIELDS = ('checked_cells', 'eligible_cells',
    'attempted_cells', 'available_cells', 'unavailable_cells', 'initial_partial_cells',
    'ineligible_cells', 'source_price_observations', 'source_volume_observations',
    'volume_rank_observations', 'descriptive_time_slope_estimates',
    'unweighted_slope_solves', 'weighted_slope_solves', 'constant_volume_copies',
    'constant_price_shortcuts', 'constant_price_cells', 'constant_volume_cells',
    'cross_sectional_rank_observations')
FEATURE_FLAGS = ('passed', 'independent_scalar_rebuild', 'official_session_windows_checked',
    'fixed_253_prices_and_volumes_checked', 'skip42_checked',
    'positive_finite_common_sources_checked', 'eligible_only_consumption_checked',
    'same_window_control_checked', 'within_window_average_volume_ranks_checked',
    'bounded_positive_weights_checked', 'log_price_relative_path_checked',
    'centered_wls_checked', 'identical_ols_control_checked',
    'constant_volume_exact_copy_checked', 'constant_price_zero_checked',
    'common_availability_checked', 'initial_partial_unchanged', 'audit_checked',
    'descriptive_counts_checked', 'no_predictive_fits_checked', 'no_future_inputs',
    'no_source_replacements')
MODEL_FLAGS = FEATURE_FLAGS
MODEL_COUNT_FIELDS = COUNT_FIELDS
SELECTION_FLAGS = ('passed', 'independent_original_pre_checked', 'original_support_checked',
    'original_budget_checked', 'symmetric_common_eligible_ranks_checked',
    'additive_pre_correction_checked', 'original_effects_unchanged',
    'one_episode_per_arm', 'initial_partial_unchanged', 'exact_scores_and_choices_checked')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_source_oracle():
    require(sha(SOURCE_HELPER) == SOURCE_SHA, 'Frozen independent source changed')
    spec = importlib.util.spec_from_file_location('volume232_original_independent', SOURCE_HELPER)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def dependency_files():
    return tuple(sorted(set([Path(__file__).resolve(), ACCOUNTING,
        *frozen_source_oracle().source_files(ROOT)])))


def accounting():
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Independent accounting source changed')
    spec = importlib.util.spec_from_file_location('volume232_independent_accounts', ACCOUNTING)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def record_check(actual, expected, label='audit'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ': keys')
        for key in expected:
            record_check(actual[key], expected[key], label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list')
        for i, (a, b) in enumerate(zip(actual, expected)):
            record_check(a, b, label + '/' + str(i))
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': value/type')


def exact_array(actual, expected, label):
    a, e = np.asarray(actual), np.asarray(expected)
    require(a.shape == e.shape and a.dtype == e.dtype, label + ': shape/dtype')
    if e.dtype.kind == 'f':
        require(not np.isinf(a).any() and np.array_equal(np.isnan(a), np.isnan(e)), label + ': finite mask')
        good = np.isfinite(e)
        require(a[good].tobytes() == e[good].tobytes(), label + ': exact binary64 values')
    else:
        require(np.array_equal(a, e), label + ': values')


def close(actual, expected, label, atol=0.):
    a, e = np.asarray(actual), np.asarray(expected)
    require(a.shape == e.shape and np.array_equal(np.isfinite(a), np.isfinite(e))
        and np.array_equal(np.isnan(a), np.isnan(e)) and not np.isinf(a).any(), label+': array domain')
    good = np.isfinite(e)
    require(np.all(np.abs(a[good]-e[good]) <= atol), label+': numerical values')


def finite(value, label):
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite ' + label)
    return value


def total(values, label):
    try:
        return finite(math.fsum(values), label)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Nonfinite ' + label) from exc


def iso(value):
    require(isinstance(value, str), 'Canonical ISO date string required')
    parsed = date.fromisoformat(value)
    require(parsed.isoformat() == value, 'Canonical ISO date string required')
    return pd.Timestamp(parsed)


def schema(panel, meta, calendar, eligible, limit):
    require(isinstance(panel, dict) and set(panel) == {'adj', 'volume'}, 'Two source frames required')
    for role, frame in panel.items():
        require(isinstance(frame, pd.DataFrame) and isinstance(frame.index, pd.DatetimeIndex), role+': frame type')
        ix = frame.index
        require(not ix.hasnans and ix.tz is None and ix.is_unique and ix.is_monotonic_increasing
            and ix.equals(ix.normalize()), role+': dates')
        require(frame.columns.is_unique and all(isinstance(s,str) and s and s.strip()==s
            for s in frame.columns), role+': identities')
    require(isinstance(meta,dict) and isinstance(meta.get('dates'),list)
        and isinstance(meta.get('symbols'),list) and len(meta['dates'])==len(meta['symbols']), 'Query axes')
    etfs=meta.get('etfs')
    require(isinstance(etfs,list) and all(isinstance(s,str) and s and s.strip()==s for s in etfs)
        and len(set(etfs))==len(etfs),'ETF identity metadata')
    e = np.asarray(eligible)
    require(e.ndim==2 and e.dtype==np.dtype(bool) and e.shape[0]==len(meta['dates']), 'Eligible bool matrix')
    rows,width=e.shape
    require(limit is None or type(limit) is int and 0<=limit<=rows, 'Prefix limit')
    n=rows if limit is None else limit
    sessions=[iso(s) for s in calendar]
    require(sessions and all(s.weekday()<5 for s in sessions)
        and all(a<b for a,b in zip(sessions,sessions[1:])), 'Official calendar')
    pos={d:i for i,d in enumerate(sessions)};queries=[]
    for i,(ds,names) in enumerate(zip(meta['dates'][:n],meta['symbols'][:n])):
        d=iso(ds)
        require(d in pos and (not queries or queries[-1][0]<d), 'Official increasing query dates')
        require(isinstance(names,(list,tuple)) and len(names)==width and width>0
            and len(set(names))==width and all(isinstance(s,str) and s and s.strip()==s for s in names), 'Query identities')
        t=pos[d]
        if ds=='2016-08-17': require(i==0, 'Initial partial must be first')
        else: require(t+1<len(sessions) and sessions[t+1].to_period('M')!=d.to_period('M'), 'Official month end')
        queries.append((d,list(names),t))
    return sessions,queries,e[:n],width


def numeric_path(raw, role):
    if any(not isinstance(v,numbers.Real) or isinstance(v,(bool,np.bool_)) for v in raw):
        return None,'nonnumeric_'+role
    try: values=[float(v) for v in raw]
    except (ValueError,TypeError,OverflowError): return None,'invalid_'+role
    if any(not math.isfinite(v) or v<=0. for v in values): return None,'invalid_'+role
    return values,None


def average_ranks(values):
    require(all(math.isfinite(v) for v in values), 'Finite ranking values')
    # Pair counting is independent of producer sorting/dataframe rank routines.
    return [1.+sum(other<v for other in values)+(sum(other==v for other in values)-1)/2.
        for v in values]


def slope_stats(x, y, w):
    W=total(w,'weight sum')
    require(W>0., 'Positive weight mass')
    xb=finite(total([finite(a*b,'weighted x') for a,b in zip(w,x)],'weighted x sum')/W,'weighted x mean')
    yb=finite(total([finite(a*b,'weighted y') for a,b in zip(w,y)],'weighted y sum')/W,'weighted y mean')
    terms=[];squares=[]
    try:
        for a,b,c in zip(w,x,y):
            dx=finite(b-xb,'time deviation');dy=finite(c-yb,'log price deviation')
            terms.append(finite((a*dx)*dy,'weighted cross product'))
            squares.append(finite(a*(dx**2),'weighted time square'))
    except OverflowError as exc: raise FloatingPointError('Nonfinite weighted slope arithmetic') from exc
    numerator=total(terms,'slope numerator');denominator=total(squares,'slope denominator')
    require(denominator>0.,'Strictly positive time variance')
    value=finite(numerator/denominator,'slope')
    return dict(weight_sum=W,x_mean=xb,y_mean=yb,numerator=numerator,denominator=denominator,slope=value)


def measure(prices, volumes):
    require(len(prices)==len(volumes)==253,'253 paired source observations')
    p,pr=numeric_path(prices,'price');v,vr=numeric_path(volumes,'volume')
    require(pr is None and vr is None,'Positive numeric sources')
    ranks=average_ranks(v);w=[.5+(r-1.)/252. for r in ranks]
    y=[finite(math.log(q)-math.log(p[0]),'relative log price') for q in p]
    flatp=all(q==p[0] for q in p);flatv=all(q==v[0] for q in v)
    if flatp:
        cs=ws=None;control=candidate=0.;cb=wb='constant_price_zero'
    else:
        x=[i/252. for i in range(253)]
        cs=slope_stats(x,y,[1.]*253);control=cs['slope'];cb='unweighted_slope'
        if flatv:
            ws=dict(cs);candidate=control;wb='constant_volume_copy'
        else:
            ws=slope_stats(x,y,w);candidate=ws['slope'];wb='weighted_slope'
    return dict(prices=p,volumes=v,volume_ranks=ranks,weights=w,log_prices_relative=y,
        constant_price=flatp,constant_volume=flatv,control_stats=cs,candidate_stats=ws,
        control=control,candidate=candidate,control_branch=cb,candidate_branch=wb)


def independent_features(panel, meta, calendar, eligible, limit=None):
    sessions,queries,E,width=schema(panel,meta,calendar,eligible,limit)
    n=len(queries);values={'candidate':np.full((n,width),np.nan),
        'control':np.full((n,width),np.nan),'available':np.zeros((n,width),bool)}
    counts={k:0 for k in COUNT_FIELDS};counts.update(events=n,symbols_per_event=width,
        checked_cells=n*width,eligible_cells=int(E.sum()))
    events=[];reasons={}
    for i,(d,names,t) in enumerate(queries):
        window=sessions[t-294:t-41] if t>=294 else []
        texts=[str(x.date()) for x in window]
        missingp=[str(x.date()) for x in window if x not in panel['adj'].index]
        missingv=[str(x.date()) for x in window if x not in panel['volume'].index]
        cells=[]
        for j,name in enumerate(names):
            cell=dict(symbol=name,date=str(d.date()),column=j,eligible=bool(E[i,j]),available=False,
                reason=None,candidate=None,control=None,prices=[],volumes=[],volume_ranks=[],weights=[],
                log_prices_relative=[],constant_price=None,constant_volume=None,
                control_stats=None,candidate_stats=None,control_branch=None,candidate_branch=None)
            if str(d.date())=='2016-08-17':
                reason='initial_partial';counts['initial_partial_cells']+=1
            elif not E[i,j]: reason='ineligible';counts['ineligible_cells']+=1
            else:
                counts['attempted_cells']+=1
                if not window: reason='insufficient_history'
                elif name not in panel['adj'].columns: reason='missing_price_symbol'
                elif name not in panel['volume'].columns: reason='missing_volume_symbol'
                elif missingp: reason='missing_price_session'
                elif missingv: reason='missing_volume_session'
                else:
                    p,reason=numeric_path([panel['adj'].at[z,name] for z in window],'price')
                    if reason is None:
                        v,reason=numeric_path([panel['volume'].at[z,name] for z in window],'volume')
                    if reason is None:
                        z=measure(p,v);cell.update(z);cell['available']=True;reason='available'
                        for key in ('candidate','control'): values[key][i,j]=z[key]
                        values['available'][i,j]=True;counts['available_cells']+=1
                        counts['constant_price_cells']+=int(z['constant_price'])
                        counts['constant_volume_cells']+=int(z['constant_volume'])
                        counts['constant_price_shortcuts']+=int(z['constant_price'])
                        counts['unweighted_slope_solves']+=int(not z['constant_price'])
                        counts['weighted_slope_solves']+=int(not z['constant_price'] and not z['constant_volume'])
                        counts['constant_volume_copies']+=int(not z['constant_price'] and z['constant_volume'])
            cell['reason']=reason;reasons[reason]=reasons.get(reason,0)+1;cells.append(cell)
        events.append(dict(date=str(d.date()),symbols=names,session_position=t,
            source_start=texts[0] if texts else None,source_end=texts[-1] if texts else None,
            source_dates=texts,missing_price_sessions=missingp,missing_volume_sessions=missingv,cells=cells))
    a=counts['available_cells'];counts['unavailable_cells']=n*width-a
    for k in ('source_price_observations','source_volume_observations','volume_rank_observations'): counts[k]=253*a
    counts['descriptive_time_slope_estimates']=2*a
    audit=dict(schema='volume_slope232_features_v1',**counts,window_price_count=253,window_volume_count=253,
        span_sessions=252,skip_sessions=42,source_start_offset=-294,source_end_offset=-42,
        reason_counts=dict(sorted(reasons.items())),common_availability=True,feature_domain='original_eligible',
        within_volume_ranking_performed=True,cross_sectional_ranking_performed=False,
        trading_eligibility_modified=False,predictive_model_fits=0,new_model_fits=0,
        portfolio_returns_calculated=False,event_history=events)
    return values,audit


def verify_features(panel,meta,calendar,eligible,values,audit,limit=None):
    expected,record=independent_features(panel,meta,calendar,eligible,limit)
    require(isinstance(values,dict) and set(values)==set(VALUE_KEYS),'Three feature arrays')
    for key in VALUE_KEYS: exact_array(values[key],expected[key],key)
    record_check(audit,record)
    return dict(**{k:True for k in FEATURE_FLAGS},counts={k:record[k] for k in COUNT_FIELDS},
        query_rows=record['events'],numerical_tolerance=0.,new_model_fits=0,predictive_model_fits=0,
        producer_modules_imported=False,computes_returns=False)


def centered_rank(values):
    v=[float(z) for z in values]
    require(all(math.isfinite(z) for z in v),'Finite rank population')
    if len(v)<2 or all(z==v[0] for z in v): return np.zeros(len(v),dtype=np.float64)
    return np.asarray([(r-1.)/(len(v)-1.)-.5 for r in average_ranks(v)],dtype=np.float64)


def ranked_features(values, eligible, dates):
    E=np.asarray(eligible);require(E.ndim==2 and E.dtype==bool,'Selection eligibility')
    require(isinstance(values,dict) and set(values)==set(VALUE_KEYS),'Feature value keys')
    mask=np.asarray(values['available'])
    require(mask.shape==E.shape and mask.dtype==bool and not np.any(mask&~E),'Common eligible feature support')
    require(len(dates)==len(E),'Selection dates')
    ranks={k:np.zeros(E.shape) for k in ('candidate','control')}
    for k in ranks:
        raw=np.asarray(values[k]);require(raw.shape==E.shape and raw.dtype==np.float64
            and not np.isinf(raw).any() and np.array_equal(np.isfinite(raw),mask),'Common finite raw slopes')
        for i,d in enumerate(dates):
            if d=='2016-08-17': require(not mask[i].any(),'Initial feature must be unavailable')
            ranks[k][i,mask[i]]=centered_rank(raw[i,mask[i]])
    return ranks


def selection_oracle(original,meta,values):
    helper=frozen_source_oracle().old.original_helper()
    x=helper.original_prefix(original,meta,None)
    ranks=ranked_features(values,x['eligible'],meta['dates'])
    budget=np.minimum(.05,np.maximum(0.,.10-(.025*x['calm']+.05*x['weak'])))
    scores={};choices={}
    for arm in ARMS:
        pre=x['local'].copy()
        if arm!=ARMS[0]:
            rank=ranks['candidate' if arm==ARMS[1] else 'control'];use=rank!=0.
            pre[use]+=.01*rank[use]
        score=pre.copy();rows=[]
        for i,names in enumerate(x['names']):
            order=sorted(np.flatnonzero(x['eligible'][i]).tolist(),key=lambda j:(-pre[i,j],names[j]))
            if len(order)<3:
                rows.append(order if len(order)==2 else []);continue
            js=[j for j in order[2:5] if float(pre[i,order[2]])-float(pre[i,j])<=.02]
            for j in js:
                target=min(.5,max(-.5,x['support'][i,j]+x['effects'][i,j]))
                score[i,j]+=budget[i]*(target-x['support'][i,j])
            third=min(js,key=lambda j:(-score[i,j],names[j]))
            rows.append(sorted(order[:2]+[third],key=lambda j:(-score[i,j],names[j])))
        scores[arm]=score;choices[arm]=rows
    return scores,choices


def verify_selection(original,meta,values,scores,choices):
    require(set(scores)==set(choices)==set(ARMS),'Three selection arms')
    expected,ordered=selection_oracle(original,meta,values)
    for arm in ARMS:
        exact_array(scores[arm],expected[arm],arm+' additive pre plus one original episode')
        require(choices[arm]==ordered[arm],arm+' ordered choices')
    ranks=ranked_features(values,np.asarray(original['eligible']),meta['dates'])
    bound=max((float(np.max(np.abs(.01*v))) if v.size else 0. for v in ranks.values()),default=0.)
    require(bound<=.005,'Fixed auxiliary maximum absolute delta')
    return dict(**{k:True for k in SELECTION_FLAGS},ordered_choices_checked=3*len(meta['dates']),
        max_absolute_delta=bound,numerical_tolerance=0.,original_source_parity_tolerance=1e-12,
        producer_modules_imported=False,computes_returns=False)


def audit_outcomes(px, opening, orders, exports, results, meta):
    helper = accounting()
    require(set(orders) == set(exports) == set(results) == set(ARMS), 'Outcome arm names differ')
    require(len(meta['dates']) == len(meta['ends']) == 122 and meta['ends'][:-1] == meta['dates'][1:], 'Outcome interval axis differs')
    require(meta['dates'][0] == '2016-08-17' and meta['ends'][-1] == '2026-09-24'
            and meta['ends'][52] == meta['dates'][53] == '2020-12-31'
            and meta['ends'][87] == meta['dates'][88] == '2023-11-30', 'Outcome fixed slices differ')
    reports = {}
    for arm in ARMS:
        require(len(orders[arm]) == 122 and [o['date'] for o in orders[arm]] == meta['dates'] and
                [o['end'] for o in orders[arm]] == meta['ends'], 'Order axis differs')
        require(set(exports[arm]) == set(results[arm]) == {p[0] for p in PATHS}, 'Execution paths differ')
        reports[arm] = {}
        for name, execution, cost in PATHS:
            data = exports[arm][name]
            require(set(data) == {'DAILY', 'TRADES', 'HOLDINGS', 'DAILY_POSITIONS'}, 'Incomplete execution exports')
            audited = helper.audit_path(px, opening, orders[arm], *[data[k] for k in ('DAILY', 'TRADES', 'HOLDINGS', 'DAILY_POSITIONS')], execution, cost)
            slices = helper.statistics_by_split(audited['days'], audited['values'], meta)
            wanted = results[arm][name]
            require(set(wanted['slices']) == set(slices), 'Fixed five slices differ')
            helper.check_metrics(slices['full122'], wanted['metrics'], arm + '/' + name)
            for split, value in slices.items():
                helper.check_metrics(value, wanted['slices'][split], arm + '/' + name + '/' + split)
            for key in ('fees_paid', 'dollar_turnover', 'trade_rows'):
                helper.near(audited[key], wanted['metrics'][key], key)
            annual = []
            for year in sorted({d[:4] for d in audited['days']}):
                ids = [i for i, day in enumerate(audited['days']) if day.startswith(year)]
                if len(ids) < 2:
                    continue
                prior = audited['values'][ids[0] - 1] if ids[0] else 100000.
                annual.append(dict(year=int(year), return_=audited['values'][ids[-1]] / prior - 1., partial=int(year) in (2016, 2026)))
            observed = wanted['metrics']['annual_returns']
            require(isinstance(observed, list) and len(observed) == len(annual), 'Annual return count')
            for actual, expected in zip(observed, annual):
                require(set(actual) == set(expected) and actual['year'] == expected['year'] and
                        actual['partial'] == expected['partial'], 'Annual return identity')
                close(actual['return_'], expected['return_'], 'Annual return', atol=1e-12)
            require(wanted['accounting']['exported_fill_ledger_matches'] is True and
                    wanted['accounting']['terminal_liquidation'] is False, 'Account declaration differs')
            for key in ('days', 'values', 'daily_cash_nav'):
                audited.pop(key)
            reports[arm][name] = dict(audited, passed=True, slices=slices)
    return dict(passed=True, checked_paths=9, checked_slices=45, independent_accounts=True,
                independent_of_production_model_and_engine=True, terminal_liquidation=False,
                accounting_source_sha256=ACCOUNTING_SHA, paths=reports)
