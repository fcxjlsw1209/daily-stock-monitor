"""Independent historical-state volatility CDF audit for round188.

No production selector/model/engine is used. The pinned pandas whole-prefix
rolling statistic preserves exact CDF ties; a cached scalar sample-standard-
deviation calculation checks each distinct consumed price window independently.
"""
from datetime import date
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTCOME_HELPER = ROOT / 'work/price_distribution171_independent.py'
OUTCOME_SHA = '94e2a984affad28ae1ba62de245519ffd9f4763c1a6ab3a5cd95dea02e0a1841'
ACCOUNTING = ROOT / 'work/ptr169_results_independent.py'
ACCOUNTING_SHA = '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59'
ARMS = ('incumbent_local67', 'state_conditioned_vol_cdf', 'unconditional_vol_cdf')
RAW = dict(state_conditioned_vol_cdf='candidate', unconditional_vol_cdf='control')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
DOSE = .01
HISTORY = 756
OWN = 126
MARKET = 63
MIN_JOINT = 252
MIN_MATCHED = 63
CALM = .10


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_files():
    return (Path(__file__).resolve(), OUTCOME_HELPER, ACCOUNTING)


def accounting():
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('state188_accounting', ACCOUNTING)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def audit_outcomes(px, opening, orders, exports, results, meta):
    require(sha(OUTCOME_HELPER) == OUTCOME_SHA, 'Frozen independent outcome helper changed')
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('state188_isolated_outcome', OUTCOME_HELPER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    require(module.ARMS == ('incumbent_local67', 'price_activity_distribution', 'price_only_distribution'), 'Unexpected original outcome arms')
    module.ARMS = ARMS
    result = module.audit_outcomes(px, opening, orders, exports, results, meta)
    require(set(result['paths']) == set(ARMS), 'Independent outcome arm set')
    require(all(set(result['paths'][a]) == {p[0] for p in PATHS} for a in ARMS), 'Independent outcome paths')
    require(result['checked_paths'] == 9 and result['checked_slices'] == 45, 'Independent outcome counts')
    result['outcome_helper_sha256'] = OUTCOME_SHA
    return result


def close(actual, expected, label, atol=1e-12):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(actual.shape == expected.shape, label + ': shape')
    require(np.array_equal(np.isfinite(actual), np.isfinite(expected)), label + ': finite mask')
    require(np.array_equal(np.isnan(actual), np.isnan(expected)), label + ': NaNs')
    require(not np.isinf(actual).any() and not np.isinf(expected).any(), label + ': infinity')
    valid = np.isfinite(expected)
    require(np.all(np.abs(actual[valid] - expected[valid]) <= atol), label + ': values')


def record_check(actual, expected, label='audit'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict), label + ': mapping')
        for key, value in expected.items():
            require(key in actual, label + ': missing ' + key)
            record_check(actual[key], value, label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list')
        for i, (got, wanted) in enumerate(zip(actual, expected)):
            record_check(got, wanted, label + '/' + str(i))
    elif isinstance(expected, float):
        require(type(actual) in (int, float), label + ': scalar')
        close(actual, expected, label)
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': differs')


def iso(value):
    require(isinstance(value, str), 'ISO date string required')
    day = date.fromisoformat(value)
    require(day.isoformat() == value, 'Strict ISO date required')
    return pd.Timestamp(day)


def schema(panel, meta, calendar):
    require(isinstance(panel, dict) and set(panel) == {'adj'}, 'Panel keys differ')
    p = panel['adj']
    require(isinstance(p, pd.DataFrame) and isinstance(p.index, pd.DatetimeIndex) and len(p), 'Price frame required')
    require(not p.index.hasnans and p.index.is_unique and p.index.is_monotonic_increasing and
            p.index.tz is None and p.index.equals(p.index.normalize()), 'Invalid source date axis')
    require(p.columns.is_unique and
            all(isinstance(s,str) and s and s.strip()==s for s in p.columns), 'Source identities')
    require(isinstance(meta, dict) and all(k in meta for k in ('dates','symbols','etfs')), 'Metadata fields')
    require(len(meta['dates'])==len(meta['symbols']), 'Metadata event lengths')
    require(isinstance(meta['etfs'], (list,tuple,set,frozenset)) and len(meta['etfs'])==len(set(meta['etfs'])) and
            all(isinstance(s,str) and s and s.strip()==s for s in meta['etfs']), 'ETF identities')
    days = [iso(s) for s in calendar]
    require(days and all(x.weekday()<5 for x in days) and all(a<b for a,b in zip(days,days[1:])), 'Calendar sessions/order')
    positions = {day:i for i,day in enumerate(days)}
    events, width = [], 0
    for text,names in zip(meta['dates'],meta['symbols']):
        day=iso(text)
        require(day in positions and (not events or day>events[-1][0]), 'Event session/order')
        require(isinstance(names,(list,tuple)) and names and len(names)==len(set(names)) and
                all(isinstance(s,str) and s and s.strip()==s for s in names), 'Event identities')
        require(not events or width==len(names), 'Changing event width')
        width=len(names); events.append((day,list(names),positions[day]))
    return days,events,width



def finite(value):
    return math.isfinite(float(value))


def nullable(value):
    return float(value) if finite(value) else None


def canonical_source(panel,queries,sessions):
    """Use the full official calendar prefix, with no price filling/compression."""
    last=queries[-1][2]
    needed={'SPY'}|{s for _,ss,_ in queries for s in ss}
    columns=[s for s in panel['adj'].columns if s in needed]
    axis=pd.DatetimeIndex(sessions[:last+1])
    # Reindex before numeric conversion: future and off-calendar cells are not read.
    try:frame=panel['adj'].reindex(index=axis,columns=columns).astype(float)
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError('Nonnumeric source price prefix') from exc
    if 'SPY' not in frame:frame['SPY']=np.nan
    raw=frame.to_numpy();frame=frame.where(np.isfinite(raw)&(raw>0.))
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        returns=frame.pct_change(fill_method=None)
    pair=frame.notna()&frame.shift(1).notna()
    if not np.isfinite(returns.to_numpy()[pair.to_numpy()]).all():
        raise FloatingPointError('Finite positive prices caused return arithmetic overflow')
    with np.errstate(over='ignore',invalid='ignore'):
        own=returns.rolling(OWN,min_periods=OWN).std(ddof=1)
        market=returns.SPY.rolling(MARKET,min_periods=MARKET).std(ddof=1)
    own_complete=returns.notna().rolling(OWN).sum().eq(OWN)
    market_complete=returns.SPY.notna().rolling(MARKET).sum().eq(MARKET)
    if not np.isfinite(own.to_numpy()[own_complete.to_numpy()]).all() or not np.isfinite(market.to_numpy()[market_complete.to_numpy()]).all():
        raise FloatingPointError('Complete finite-return window caused volatility overflow')
    annual=market*np.sqrt(252.)
    if not np.isfinite(annual.to_numpy()[market.notna().to_numpy()]).all():
        raise FloatingPointError('Annualized market volatility overflow')
    # Crucially, missing risk remains NaN; .lt() alone would turn it into False.
    states=pd.Series(np.nan,index=axis)
    ok=np.isfinite(annual.to_numpy())
    states.iloc[np.flatnonzero(ok)]=(annual.iloc[np.flatnonzero(ok)]<CALM).astype(float)
    return frame,own,market,annual,states


def sample_sigma(prices):
    """Raw-price scalar oracle, independent of pandas' streaming variance."""
    if any(not finite(p) or float(p)<=0. for p in prices):return math.nan
    ret=[]
    for a,b in zip(prices,prices[1:]):
        value=float(b)/float(a)-1.
        if not finite(value):raise FloatingPointError('Scalar return overflow')
        ret.append(value)
    try:
        mean=math.fsum(ret)/len(ret)
        squared=math.fsum((r-mean)*(r-mean) for r in ret)
        sigma=math.sqrt(squared/(len(ret)-1))
    except (OverflowError,ValueError) as exc:raise FloatingPointError('Scalar sample-volatility overflow') from exc
    if not finite(sigma):raise FloatingPointError('Scalar sample-volatility overflow')
    return sigma


def cdf_oracle(current,current_calm,history,states,history_complete=True,missing_symbol=False):
    """Scalar midpoint counts over fixed slots; never interpret absent state as 0."""
    require(type(history_complete) is bool and type(missing_symbol) is bool,'Boolean structure flags')
    require(current_calm is None or type(current_calm) is bool,'Current state must be bool or missing')
    require(len(history)==len(states)==(HISTORY if history_complete else 0),'Exact historical slot count')
    joint=[];matched=[]
    for sigma,state in zip(history,states):
        if state is not None:require(type(state) is bool,'Historical state must be bool or missing')
        if sigma is not None and finite(sigma):
            require(float(sigma)>=0.,'Negative volatility')
            if state is not None:
                joint.append(float(sigma))
                if current_calm is not None and state==current_calm:matched.append(float(sigma))
    present=current is not None and finite(current)
    if present:require(float(current)>=0.,'Negative current volatility')
    rb=re=mb=me=None
    if present and history_complete:
        rb=sum(x<float(current) for x in joint);re=sum(x==float(current) for x in joint)
        if current_calm is not None:
            mb=sum(x<float(current) for x in matched);me=sum(x==float(current) for x in matched)
    reason=('insufficient_history' if not history_complete else 'missing_symbol' if missing_symbol else
            'current_market_state_unavailable' if current_calm is None else
            'current_volatility_unavailable' if not present else 'insufficient_joint_history' if len(joint)<MIN_JOINT else
            'insufficient_matched_history' if len(matched)<MIN_MATCHED else 'available')
    ready=reason=='available'
    candidate=(mb+.5*me)/len(matched)-.5 if ready else None
    control=(rb+.5*re)/len(joint)-.5 if ready else None
    return dict(current_sigma=float(current) if present else None,current_calm=current_calm,
                reference_count=len(joint),matched_count=len(matched),reference_below=rb,reference_equal=re,
                matched_below=mb,matched_equal=me,candidate=candidate,control=control,available=ready,reason=reason)


def independent_features(panel,meta,calendar,limit=None,checks=None):
    sessions,queries,width=schema(panel,meta,calendar)
    require(limit is None or type(limit) is int and 0<=limit<=len(queries),'Invalid feature prefix')
    queries=queries if limit is None else queries[:limit];n=len(queries)
    values=dict(candidate=np.full((n,width),np.nan),control=np.full((n,width),np.nan),available=np.zeros((n,width),bool))
    rows=[];reasons={};references=matched_total=0;cache={};scalar_checks=0
    source=canonical_source(panel,queries,sessions) if queries else None
    if source is not None:
        frame,own,market,annual,states=source
        price_arrays={s:frame[s].to_numpy(float) for s in frame}
        own_arrays={s:own[s].to_numpy(float) for s in own}
        market_values=market.to_numpy(float);state_values=states.to_numpy(float)

    def checked(symbol,t,window):
        nonlocal scalar_checks
        key=(symbol,t,window)
        if key in cache:return cache[key]
        can=(market_values[t] if window==MARKET else own_arrays[symbol][t]) if symbol in price_arrays else math.nan
        prices=price_arrays[symbol][t-window:t+1] if symbol in price_arrays and t>=window else np.array([])
        scalar=sample_sigma(prices) if len(prices)==window+1 else math.nan
        require(finite(can)==finite(scalar),'Canonical rolling support differs from original prices')
        if finite(can):
            close(can,scalar,'Independent scalar volatility',1e-12+1e-10*abs(scalar))
            scalar_checks+=1
        cache[key]=float(can)
        return cache[key]

    for day,names,t in queries:
        complete=t>=HISTORY
        history_positions=range(t-HISTORY,t) if complete else range(0)
        current_market=checked('SPY',t,MARKET)
        current_calm=bool(state_values[t]) if finite(state_values[t]) else None
        reference_states=[]
        for q in history_positions:
            historical_market=checked('SPY',q,MARKET)
            s=bool(state_values[q]) if finite(historical_market) else None
            # Independently reconstruct each canonical calm comparison.
            if s is not None:require(s==(historical_market*math.sqrt(252.)<CALM),'Historical calm state differs')
            reference_states.append(s)
        if current_calm is not None:require(current_calm==(current_market*math.sqrt(252.)<CALM),'Current calm state differs')
        start=str(sessions[t-HISTORY].date()) if complete else None
        end=str(sessions[t-1].date()) if complete else None
        cells=[]
        for j,symbol in enumerate(names):
            missing=symbol not in panel['adj'].columns
            current=checked(symbol,t,OWN)
            history=[checked(symbol,q,OWN) for q in history_positions]
            cell=cdf_oracle(nullable(current),current_calm,history,reference_states,complete,missing)
            cell=dict(symbol=symbol,date=str(day.date()),**cell,reference_start=start,reference_end=end)
            if cell['available']:
                for key in ('candidate','control'):values[key][len(rows),j]=cell[key]
                values['available'][len(rows),j]=True
            references+=cell['reference_count'];matched_total+=cell['matched_count']
            reasons[cell['reason']]=reasons.get(cell['reason'],0)+1;cells.append(cell)
        consumed=sessions[max(0,t-HISTORY-OWN):t+1]
        rows.append(dict(date=str(day.date()),symbols=names,session_position=t,current_calm=current_calm,
            spy_sigma63=nullable(current_market),spy_annualized_sigma=nullable(annual.iloc[t]),
            reference_start=start,reference_end=end,reference_slot_count=HISTORY if complete else 0,
            own_price_start=str(sessions[t-OWN].date()) if t>=OWN else None,own_price_end=str(day.date()),
            market_price_start=str(sessions[t-MARKET].date()) if t>=MARKET else None,market_price_end=str(day.date()),
            source_start=str(consumed[0].date()),source_end=str(day.date()),
            missing_official_sessions=[str(d.date()) for d in consumed if d not in panel['adj'].index],cells=cells))
    result=dict(events=n,symbols_per_event=width,checked_cells=n*width,history_slots=HISTORY,own_return_count=OWN,
        market_return_count=MARKET,minimum_joint=MIN_JOINT,minimum_matched=MIN_MATCHED,calm_threshold=CALM,
        annualization=252,ddof=1,available_cells=int(values['available'].sum()),unavailable_cells=int((~values['available']).sum()),
        reference_observations=references,matched_observations=matched_total,reason_counts=reasons,predictive_model_fits=0,
        common_availability=True,portfolio_returns_calculated=False,ranking_performed=False,trading_eligibility_modified=False,event_history=rows)
    if checks is not None:
        checks.update(scalar_volatility_checks=scalar_checks,unique_volatility_windows_checked=len(cache))
    return values,result


def arrays(values):
    require(isinstance(values,dict) and set(values)=={'candidate','control','available'},'Feature keys')
    out={k:np.asarray(v) for k,v in values.items()};mask=out['available']
    require(mask.ndim==2 and mask.dtype==bool,'Boolean availability matrix')
    for key in ('candidate','control'):
        a=out[key]
        require(a.shape==mask.shape and np.issubdtype(a.dtype,np.floating),'Raw feature shape/dtype')
        require(not np.isinf(a).any() and np.array_equal(np.isfinite(a),mask),'Common feature availability')
        require(np.all(np.abs(a[mask])<=.5),'Raw CDF must be within [-.5,.5]')
    return out


def verify_features(panel,meta,calendar,values,audit):
    actual=arrays(values);checks={}
    expected,record=independent_features(panel,meta,calendar,checks=checks)
    require(np.array_equal(actual['available'],expected['available']),'Independent common support mismatch')
    for key in ('candidate','control'):
        # Counts and rational CDF formula have no reduction-order ambiguity.
        require(np.array_equal(actual[key],expected[key],equal_nan=True),'Exact independently counted midpoint CDF differs: '+key)
    record_check(audit,record)
    keys=('events','symbols_per_event','checked_cells','available_cells','unavailable_cells','reference_observations','matched_observations')
    return dict(passed=True,**{k:record[k] for k in keys},**checks,independent_scalar_rebuild=True,
        excludes_current_reference=True,market_state_validity_checked=True,midpoint_cdf_checked=True,
        official_session_windows_checked=True,own126_volatility_checked=True,spy63_calm_state_checked=True,previous756_excludes_current_checked=True,
        joint_reference_support_checked=True,same_state_reference_checked=True,midpoint_cdf_exact_ties_checked=True,
        exact_common_availability=True,audit_checked=True,no_future_or_outcome_inputs=True,no_cross_sectional_ranking=True,
        predictive_model_fits=0,computes_returns=False,portfolio_returns_calculated=False)


def independent_deltas(values,dose=DOSE):
    require(type(dose) in (int,float) and dose==DOSE,'Fixed dose .01 required')
    a=arrays(values);deltas={arm:np.zeros(a['available'].shape) for arm in RAW}
    for arm,key in RAW.items():
        for i,j in zip(*np.nonzero(a['available'])):
            if i:deltas[arm][i,j]=dose*float(a[key][i,j])
    return deltas


def verify_deltas(values,deltas,dose=DOSE):
    require(isinstance(deltas,dict) and set(deltas)==set(RAW),'Delta arms')
    a=arrays(values);expected=independent_deltas(values,dose)
    for arm,wanted in expected.items():
        actual=np.asarray(deltas[arm]);close(actual,wanted,'Independent direct CDF delta '+arm,atol=2e-17)
        require(np.isfinite(actual).all() and np.all(np.abs(actual)<=DOSE*.5),'CDF delta exceeds fixed bound')
        require(not len(actual) or np.array_equal(actual[0],np.zeros(actual.shape[1])),'Initial delta must be exactly zero')
        absent=~a['available'];require(np.array_equal(actual[absent],np.zeros(absent.sum())),'Unavailable delta must be exactly zero')
    return dict(passed=True,checked_events=len(a['available']),dose=DOSE,max_absolute_delta=DOSE*.5,
        full_candidate_universe=True,common_available_only=True,common_availability=True,initial_delta_zero=True,
        direct_additive_pre_score=True,no_secondary_rank=True,no_cross_sectional_ranking=True,
        no_eligibility_or_model_filter=True,direct_linear_mapping=True,
        changed_cells={arm:int(np.count_nonzero(x)) for arm,x in expected.items()})
