"""Independent source, eligibility, midrank and half-core oracle for round187.

No production selector/model/engine is used. The pinned pandas whole-prefix
rolling statistic preserves the original tie domain; a scalar sample-standard-
deviation calculation independently checks each requested canonical statistic.
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
ARMS = ('incumbent_local67', 'eligible_core_rank', 'affine_core_span')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
FRACTION = .5


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_files():
    return (Path(__file__).resolve(), OUTCOME_HELPER, ACCOUNTING)


def accounting():
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('core187_accounting', ACCOUNTING)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def audit_outcomes(px, opening, orders, exports, results, meta):
    require(sha(OUTCOME_HELPER) == OUTCOME_SHA, 'Frozen independent outcome helper changed')
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('core187_isolated_outcome', OUTCOME_HELPER)
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
    require(p.columns.is_unique and 'SPY' in p.columns and
            all(isinstance(s,str) and s and s.strip()==s for s in p.columns), 'Source identities/SPY missing')
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


def prefix_matrix(value,shape,n,name,boolean=False):
    a=np.asarray(value)
    require(a.shape==shape and not np.iscomplexobj(a), name+': matrix shape/type')
    if boolean:
        require(a.dtype==bool, name+': boolean required')
        return a[:n]
    try: out=np.asarray(a[:n],float)
    except (TypeError,ValueError,OverflowError) as exc: raise ValueError(name+': numeric prefix') from exc
    require(np.isfinite(out).all(), name+': nonfinite prefix')
    return out


def prefix_weight(value,shape,n,name):
    a=np.asarray(value)
    require(not np.iscomplexobj(a),name+': real coefficient')
    if a.shape==(shape[0],): a=a[:,None]
    require(a.shape in ((shape[0],1),shape),name+': coefficient shape')
    try: out=np.broadcast_to(np.asarray(a[:n],float),(n,shape[1]))
    except (TypeError,ValueError,OverflowError) as exc: raise ValueError(name+': numeric prefix') from exc
    require(np.isfinite(out).all(),name+': nonfinite prefix')
    return out


def inputs(original,source,rows,width,limit=None):
    n=rows if limit is None else limit
    require(type(n) is int and 0<=n<=rows,'Original prefix limit')
    shape=(rows,width)
    require(isinstance(original,dict) and all(k in original for k in ('core','core_weight','state_calm10','eligible')), 'Original core inputs')
    require(isinstance(source,dict) and set(source)<= {'vol','vol_weight','state_highvol','core','core_weight'}, 'Source quality inputs')
    core=prefix_matrix(original['core'],shape,n,'original core')
    cw=prefix_weight(original['core_weight'],shape,n,'core_weight')
    calm=np.asarray(original['state_calm10'])
    require(calm.shape==(rows,) and calm.dtype==bool,'calm: boolean state shape')
    calm=calm[:n]
    eligible=prefix_matrix(original['eligible'],shape,n,'eligible',True)
    require(np.array_equal(cw,np.broadcast_to(np.where(calm,.5,.52)[:,None],(n,width))), 'Original core coefficient/state mismatch')
    require(np.all(np.abs(core/cw)<=.5+1e-12),'Original core rank bounds')
    if 'core' in source: require(np.array_equal(prefix_matrix(source['core'],shape,n,'quality core'),core),'Quality weighted core parity')
    if 'core_weight' in source: require(np.array_equal(prefix_weight(source['core_weight'],shape,n,'quality core_weight'),cw),'Quality core coefficient parity')
    return core,cw,calm,eligible


def ranks(vector):
    """Original average-percentile centered rank, missing values neutral zero."""
    finite=[j for j,x in enumerate(vector) if math.isfinite(float(x))]
    result=np.zeros(len(vector))
    for j in finite:
        x=float(vector[j]); below=sum(float(vector[k])<x for k in finite); equal=sum(float(vector[k])==x for k in finite)
        result[j]=(below+(equal+1.)/2.)/len(finite)-.5
    return result


def json_values(a):
    return [float(x) if math.isfinite(float(x)) else None for x in a]


def raw_source(panel,events,sessions):
    """Whole original prefix: calendar gaps remain NaN, never compressed or filled."""
    last=events[-1][0];needed={'SPY'}|{name for _,names,_ in events for name in names}
    frame=panel['adj'].loc[:last,[s for s in panel['adj'] if s in needed]].copy(deep=True)
    require(len(frame)>0 and all(d.weekday()<5 for d in frame.index),'Source prefix weekday/history')
    official=pd.DatetimeIndex(sessions);known=frame.index[frame.index>=official[0]]
    require(known.isin(official).all(),'Source date not in official calendar')
    early=frame.index[frame.index<official[0]]
    axis=early.append(official[(official>=max(frame.index[0],official[0]))&(official<=last)])
    try: frame=frame.reindex(axis).astype(float)
    except (TypeError,ValueError,OverflowError) as exc: raise ValueError('Nonnumeric original price history') from exc
    raw=frame.to_numpy(); require(not np.isinf(raw).any() and not np.any(np.isfinite(raw)&(raw<=0.)), 'Nonpositive or infinite original price')
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        daily=frame.pct_change(fill_method=None)
        v63=daily.rolling(63).std(ddof=1)
        m252=frame.shift(42)/frame.shift(252)-1.
        m294=frame.shift(42)/frame.shift(294)-1.
        risk=m294/v63.replace(0.,np.nan)
    m252=m252.replace([np.inf,-np.inf],np.nan);risk=risk.replace([np.inf,-np.inf],np.nan)
    warm=frame.notna().rolling(295).sum().eq(295)&frame.gt(0.)
    return dict(frame=frame,daily=daily,v63=v63,core_calm=m252,core_other=risk,
                calm=v63.SPY*np.sqrt(252)<.10,warm=warm,eligible=warm&m252.gt(0.))


def eligible_oracle(core_rank,raw_valid,eligible,initial=False):
    u=np.asarray(core_rank,float);valid=np.asarray(raw_valid);e=np.asarray(eligible)
    require(u.ndim==1 and valid.shape==u.shape and e.shape==u.shape and valid.dtype==bool and e.dtype==bool,'Eligible operator shapes')
    require(np.isfinite(u).all() and np.all(np.abs(u)<=.5+1e-12),'Canonical core rank bounds')
    require(type(initial) is bool,'Initial row flag')
    ids=[j for j in range(len(u)) if e[j]];n=len(ids)
    levels=len(set(float(u[j]) for j in ids));missing=sum(not bool(valid[j]) for j in ids)
    reason=('initial_allocation' if initial else 'missing_eligible_core' if missing else
            'fewer_than_two_eligible' if n<2 else 'all_eligible' if n==len(u) else
            'equal_eligible_core' if levels==1 else 'available')
    ready=reason=='available'
    candidate=np.full(len(u),np.nan);control=np.full(len(u),np.nan)
    lo=hi=tlo=thi=slope=diff=None;two=False;equivalent=False
    if ready:
        subset=[float(u[j]) for j in ids];target=ranks(subset)
        lo=min(subset);hi=max(subset);tlo=float(min(target));thi=float(max(target));slope=(thi-tlo)/(hi-lo)
        two=levels==2
        for k,j in enumerate(ids):
            candidate[j]=target[k]
            if two: control[j]=candidate[j]
            elif u[j]==lo: control[j]=tlo
            elif u[j]==hi: control[j]=thi
            else: control[j]=tlo+(float(u[j])-lo)*(thi-tlo)/(hi-lo)
        require(np.isfinite(candidate[e]).all() and np.isfinite(control[e]).all(),'Nonfinite eligible transformation')
        diff=max(abs(float(candidate[j])-float(control[j])) for j in ids);equivalent=diff==0.
    row=dict(eligible_count=n,distinct_eligible_levels=levels,missing_eligible_core_count=missing,row_ready=ready,reason=reason,
             original_min=lo,original_max=hi,target_min=tlo,target_max=thi,affine_slope=slope,
             control_two_level_copy=two,exact_equivalent=equivalent,max_candidate_control_difference=diff)
    return candidate,control,e&ready,row


def independent_features(panel,meta,calendar,original,source,limit=None):
    sessions,events,width=schema(panel,meta,calendar)
    require(limit is None or type(limit) is int and 0<=limit<=len(events),'Invalid prefix limit')
    core,cw,calm,eligible=inputs(original,source,len(events),width,limit)
    events=events if limit is None else events[:limit];n=len(events)
    values=dict(candidate=np.full((n,width),np.nan),control=np.full((n,width),np.nan),available=np.zeros((n,width),bool))
    raw=raw_source(panel,events,sessions) if events else None
    history=[];reasons={};ready_rows=eligible_count=raw_count=equivalent_rows=different_rows=0;max_diff=0.
    for i,(day,names,pos) in enumerate(events):
        require(bool(raw['calm'].at[day])==bool(calm[i]),'Original market state parity')
        active=raw['core_calm' if calm[i] else 'core_other'].loc[day].reindex(names).to_numpy(float)
        rebuilt=ranks(active);valid=np.isfinite(active)
        close(core[i],rebuilt*cw[i],'Original weighted core parity')
        close(core[i]/cw[i],rebuilt,'Original core rank parity')
        good=raw['eligible'].loc[day].reindex(names,fill_value=False).to_numpy(bool)
        require(np.array_equal(good,eligible[i]),'Original eligibility parity')
        # Scalar verification is deliberately independent of pandas' streaming
        # variance algorithm; original bits remain the source of exact ranks.
        end=raw['daily'].index.get_loc(day)
        for name in set(names)|{'SPY'}:
            if name not in raw['daily']: continue
            window=raw['daily'][name].iloc[max(0,end-62):end+1].to_numpy(float)
            if len(window)==63 and np.isfinite(window).all() and math.isfinite(float(raw['v63'].at[day,name])):
                mean=math.fsum(float(x) for x in window)/63.
                sigma=math.hypot(*(float(x)-mean for x in window))/math.sqrt(62.)
                close(raw['v63'].at[day,name],sigma,'Independent scalar sample volatility',1e-12+1e-12*abs(sigma))
        a,b,mask,row=eligible_oracle(rebuilt,valid,good,initial=i==0)
        values['candidate'][i]=a;values['control'][i]=b;values['available'][i]=mask
        row=dict(date=str(day.date()),symbols=names,session_position=pos,calm=bool(calm[i]),core_horizon=252 if calm[i] else 294,
                 raw_core=json_values(active),core_rank=rebuilt.tolist(),core_valid=valid.tolist(),eligible=good.tolist(),
                 candidate=json_values(a),control=json_values(b),**row)
        eligible_count+=row['eligible_count'];raw_count+=int(valid.sum());ready_rows+=int(row['row_ready'])
        equivalent_rows+=int(row['row_ready'] and row['exact_equivalent'])
        different_rows+=int(row['row_ready'] and not row['exact_equivalent'])
        if row['row_ready']:max_diff=max(max_diff,row['max_candidate_control_difference'])
        reasons[row['reason']]=reasons.get(row['reason'],0)+1;history.append(row)
    audit=dict(events=n,symbols_per_event=width,checked_cells=n*width,eligible_cells=eligible_count,
               raw_core_finite_cells=raw_count,available_cells=int(values['available'].sum()),
               unavailable_cells=int((~values['available']).sum()),ready_rows=ready_rows,fallback_rows=n-ready_rows,
               equivalent_ready_rows=equivalent_rows,candidate_control_different_rows=different_rows,
               max_candidate_control_difference=max_diff,core_rank_reconstructions=n,predictive_model_fits=0,
               replacement_fraction=FRACTION,reason_counts=reasons,event_history=history,
               common_availability=True,ranking_eligibility_filtered=True,trading_eligibility_modified=False,
               portfolio_returns_calculated=False,original_core_parity_checked=True,eligibility_parity_checked=True,
               calm_state_parity_checked=True)
    return values,audit


def arrays(values,limit=None):
    require(isinstance(values,dict) and set(values)=={'candidate','control','available'},'Feature keys')
    mask=np.asarray(values['available'])
    require(mask.ndim==2 and mask.dtype==bool,'Boolean matrix support')
    n=len(mask) if limit is None else limit
    require(type(n) is int and 0<=n<=len(mask),'Feature prefix')
    out={'available':mask[:n]}
    for key in ('candidate','control'):
        a=np.asarray(values[key]);require(a.shape==mask.shape and not np.iscomplexobj(a),'Feature matrix shape/type')
        try:a=np.asarray(a[:n],float)
        except (TypeError,ValueError,OverflowError) as exc:raise ValueError('Feature numeric prefix') from exc
        require(not np.isinf(a).any() and np.array_equal(np.isfinite(a),out['available']),'Common feature availability')
        require(np.all(np.abs(a[out['available']])<=.5+1e-12),'Eligible rank bound')
        out[key]=a
    return out


def verify_features(panel,meta,calendar,original,source,values,audit):
    got=arrays(values);wanted,expected=independent_features(panel,meta,calendar,original,source)
    require(np.array_equal(got['available'],wanted['available']),'Independent feature support')
    for k in ('candidate','control'):close(got[k],wanted[k],'Independent '+k)
    record_check(audit,expected)
    for i,row in enumerate(expected['event_history']):
        if row['control_two_level_copy']:
            require(np.array_equal(got['candidate'][i],got['control'][i],equal_nan=True),'Two-level control copy must be exact')
        if row['row_ready']:
            for j,u in enumerate(row['core_rank']):
                if row['eligible'][j] and u in (row['original_min'],row['original_max']):
                    require(got['candidate'][i,j]==got['control'][i,j],'Affine endpoints must be exact')
    counts=('events','symbols_per_event','checked_cells','eligible_cells','raw_core_finite_cells','available_cells','unavailable_cells',
            'ready_rows','fallback_rows','equivalent_ready_rows','candidate_control_different_rows',
            'max_candidate_control_difference','core_rank_reconstructions')
    return dict(passed=True,**{k:expected[k] for k in counts},independent_scalar_rebuild=True,original_raw_core_checked=True,
                original_core_rank_parity_checked=True,original_market_state_checked=True,original_eligibility_checked=True,
                eligible_domain_preserved=True,conditional_midrank_checked=True,affine_span_control_checked=True,
                identity_branches_exact_checked=True,exact_common_availability=True,no_future_or_outcome_inputs=True,
                audit_checked=True,predictive_model_fits=0,computes_returns=False,portfolio_returns_calculated=False)


def independent_deltas(values,original,source,limit=None):
    a=arrays(values,limit);n,width=a['available'].shape
    shape=np.shape(original['core']);require(len(shape)==2 and shape[1]==width and n<=shape[0],'Original delta axes')
    core,cw,calm,eligible=inputs(original,source,shape[0],width,n)
    require(not np.any(a['available']&~eligible),'New support outside original eligible')
    require(not n or not a['available'][0].any(),'Initial feature support must be neutral')
    result={k:np.zeros((n,width)) for k in ('candidate','control')}
    for i in range(n):
        for j in range(width):
            if i and a['available'][i,j]:
                for key in result:
                    result[key][i,j]=FRACTION*(float(cw[i,j])*float(a[key][i,j])-float(core[i,j]))
                    require(math.isfinite(result[key][i,j]),'Nonfinite half-core delta')
    detail=dict(events=n,width=width,dose=FRACTION,initial_delta_zero=True,available_cells=int(a['available'].sum()),
                changed_cells={k:int(np.count_nonzero(x)) for k,x in result.items()},
                max_abs_delta={k:float(np.max(np.abs(x))) if x.size else 0. for k,x in result.items()},
                core_weights=cw.tolist(),core_weight_preserved=True,original_support_unchanged=True,
                original_model_corrections_unchanged=True,episode_applied=False)
    return result,detail


def verify_deltas(values,original,source,deltas,audit):
    require(isinstance(deltas,dict) and set(deltas)=={'candidate','control'},'Delta keys')
    wanted,expected=independent_deltas(values,original,source);a=arrays(values)
    for key in wanted:
        actual=np.asarray(deltas[key]);close(actual,wanted[key],'Independent half-core delta '+key,2e-17)
        neutral=~a['available'].copy()
        if len(neutral):neutral[0]=True
        require(np.array_equal(actual[neutral],np.zeros(int(neutral.sum()))),'Unavailable delta must be exactly zero')
    record_check(audit,expected)
    return dict(passed=True,events=expected['events'],symbols_per_event=expected['width'],checked_cells=int(a['available'].size),
                available_cells=expected['available_cells'],independent_scalar_rebuild=True,half_core_replacement_checked=True,
                initial_delta_zero=True,unavailable_exact_zero=True,original_support_unchanged=True,
                original_model_corrections_unchanged=True,no_extra_rank_or_model_filter=True,audit_checked=True,
                replacement_fraction=FRACTION,predictive_model_fits=0,episode_applications=0,
                changed_cells=expected['changed_cells'],audit=expected)
