"""Independent canonical-source and scalar pair/coefficient oracle for round184.

Canonical original rolling statistics use pinned pandas on the full requested
historical prefix. Ranks, symmetric comparisons and per-security coefficient
accounting are independently implemented, without importing the producer.
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
ARMS = ('incumbent_local67', 'core_peer_volatility', 'uniform_volatility_shrink_control')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
H = .20
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
    spec = importlib.util.spec_from_file_location('peer184_accounting', ACCOUNTING)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def audit_outcomes(px, opening, orders, exports, results, meta):
    require(sha(OUTCOME_HELPER) == OUTCOME_SHA, 'Frozen independent outcome helper changed')
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('peer184_isolated_outcome', OUTCOME_HELPER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    require(module.ARMS == ('incumbent_local67', 'price_activity_distribution', 'price_only_distribution'),
            'Unexpected original outcome arms')
    module.ARMS = ARMS
    result = module.audit_outcomes(px, opening, orders, exports, results, meta)
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


def vector(value, rows, name, boolean=False):
    a=np.asarray(value)
    require(a.shape==(rows,), name+': state shape')
    if boolean:
        require(a.dtype==bool,name+': boolean required')
    return a


def matrix(value, shape, name, boolean=False):
    a=np.asarray(value)
    require(a.shape==shape,name+': matrix shape')
    require(a.dtype==bool if boolean else np.issubdtype(a.dtype,np.number) and a.dtype.kind!='c',name+': dtype')
    return a


def broadcast(value, shape, name):
    a=np.asarray(value)
    require(np.issubdtype(a.dtype,np.number) and a.dtype.kind!='c',name+': numeric dtype')
    if a.shape==(shape[0],): a=a[:,None]
    require(a.shape in ((shape[0],1),shape),name+': broadcast shape')
    out=np.broadcast_to(a,shape)
    require(np.isfinite(out).all(),name+': nonfinite coefficient')
    return out


def inputs(original,source,rows,width):
    shape=(rows,width)
    require(isinstance(original,dict) and all(k in original for k in ('core','core_weight','state_calm10')), 'Original core inputs')
    require(isinstance(source,dict) and {'vol','vol_weight','state_highvol'} <= set(source) <=
            {'vol','vol_weight','state_highvol','core','core_weight'},'Source quality inputs')
    core=matrix(original['core'],shape,'original core')
    require(np.isfinite(core).all(),'Original weighted core must be finite')
    cw=broadcast(original['core_weight'],shape,'core_weight')
    calm=vector(original['state_calm10'],rows,'calm',True)
    require(np.array_equal(cw,np.broadcast_to(np.where(calm,.5,.52)[:,None],shape)), 'Original core coefficient/state mismatch')
    v=matrix(source['vol'],shape,'source vol')
    require(np.isfinite(v).all() and np.all(np.abs(v)<=.5+1e-12),'Original vol ranks must be finite bounded')
    high=vector(source['state_highvol'],rows,'highvol',True)
    base=broadcast(source['vol_weight'],shape,'vol_weight')
    expected=np.broadcast_to(np.where(calm,.2,.219375*(1-np.where(high,.10,0.)/.48))[:,None],shape)
    require(np.array_equal(base,expected),'Original volatility coefficient/state mismatch')
    if 'core' in source: require(np.array_equal(source['core'],core),'Quality weighted core parity')
    if 'core_weight' in source: require(np.array_equal(broadcast(source['core_weight'],shape,'source core_weight'),cw),'Quality core_weight parity')
    return core,cw,calm,v,base,high


def ranks(vector):
    finite=[j for j,x in enumerate(vector) if math.isfinite(float(x))]
    result=np.zeros(len(vector))
    for j in finite:
        x=float(vector[j]); below=sum(float(vector[k])<x for k in finite); equal=sum(float(vector[k])==x for k in finite)
        result[j]=(below+(equal+1.)/2.)/len(finite)-.5
    return result


def json_values(a):
    return [float(x) if math.isfinite(float(x)) else None for x in a]


def raw_source(panel,meta,calendar,last_date):
    """Reconstruct the declared original rolling algorithm; never shorten its history."""
    requested={'SPY'}
    for day,names in zip(meta['dates'],meta['symbols']):
        if iso(day)<=last_date: requested.update(names)
    columns=[name for name in panel['adj'].columns if name in requested]
    frame=panel['adj'].loc[:last_date,columns].copy(deep=True)
    require(len(frame),'No source history by query date')
    require(all(day.weekday()<5 for day in frame.index),'Nonweekday source session')
    official=pd.DatetimeIndex(calendar)
    within=frame.index[frame.index>=official[0]]
    require(within.isin(official).all(),'Source date not in official calendar')
    early=frame.index[frame.index<official[0]]
    normal=official[(official>=max(frame.index[0],official[0]))&(official<=last_date)]
    axis=early.append(normal)
    frame=frame.reindex(axis)
    try:
        frame=frame.astype(float)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError('Nonnumeric original price history') from exc
    raw=frame.to_numpy()
    require(not np.isinf(raw).any() and not np.any(np.isfinite(raw)&(raw<=0.)), 'Nonpositive or infinite original price')
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        daily=frame.pct_change(fill_method=None)
    require(not np.isinf(daily.to_numpy()).any(),'Simple-return arithmetic overflow')
    s63=daily.rolling(63).std(ddof=1)
    s126=daily.rolling(126).std(ddof=1)
    calm=(s63['SPY']*np.sqrt(252)<.10)
    high=s63['SPY']>s63['SPY'].rolling(252).median()
    core_calm=frame.shift(42)/frame.shift(252)-1.
    core_other=(frame.shift(42)/frame.shift(294)-1.)/s63.replace(0.,np.nan)
    return dict(vol=s126.replace([np.inf,-np.inf],np.nan),core_calm=core_calm.replace([np.inf,-np.inf],np.nan),
                core_other=core_other.replace([np.inf,-np.inf],np.nan),calm=calm,high=high)


def peer_oracle(raw_vol,raw_core,u,v):
    """Scalar leave-self-out symmetric comparison, retaining the full original N."""
    width=len(raw_vol)
    require(all(np.shape(x)==(width,) for x in (raw_core,u,v)),'Peer input shape')
    domain=[j for j,x in enumerate(raw_vol) if math.isfinite(float(x))]
    core_valid=[math.isfinite(float(x)) for x in raw_core]
    n=len(domain)
    reason='insufficient_vol_domain' if n<2 else 'missing_core_in_vol_domain' if any(not core_valid[j] for j in domain) else 'available'
    ready=reason=='available'
    candidate=np.full(width,np.nan);control=np.full(width,np.nan)
    neighbors=[0.]*width;signed=[0.]*width;mass=0.;nonzero=0;identity=None;mean_weight=None
    if ready:
        terms=[[] for _ in range(width)];weights=[[] for _ in range(width)];unique_weights=[]
        for a,i in enumerate(domain):
            for j in domain[a+1:]:
                w=max(1.-abs(float(u[i])-float(u[j]))/H,0.)
                sign=1. if raw_vol[i]>raw_vol[j] else -1. if raw_vol[i]<raw_vol[j] else 0.
                terms[i].append(w*sign);terms[j].append(-w*sign)
                weights[i].append(w);weights[j].append(w);unique_weights.append(w)
                nonzero+=int(w>0.)
        mass=math.fsum(unique_weights);mean_weight=2.*mass/(n*(n-1))
        for j in domain:
            neighbors[j]=math.fsum(weights[j]);signed[j]=math.fsum(terms[j])
        if all(float(u[j])==float(u[domain[0]]) for j in domain): identity='all_core_ties'
        elif all(float(raw_vol[j])==float(raw_vol[domain[0]]) for j in domain): identity='all_vol_ties'
        for j in domain:
            if identity:
                candidate[j]=control[j]=v[j]
            else:
                candidate[j]=.5/n+signed[j]/(2.*n)
                control[j]=.5/n+mean_weight*(float(v[j])-.5/n)
        require(np.isfinite(candidate[domain]).all() and np.isfinite(control[domain]).all(),'Nonfinite peer result')
    audit=dict(vol_domain=[j in domain for j in range(width)],core_valid=core_valid,domain_count=n,
               row_ready=ready,reason=reason,identity_branch=identity,mean_peer_weight=mean_weight,
               unordered_pair_weight_sum=mass,neighbor_weight_sums=neighbors,signed_comparison_sums=signed,
               candidate=json_values(candidate),control=json_values(control),
               candidate_rank_sum=math.fsum(candidate[domain]) if ready else None,
               control_rank_sum=math.fsum(control[domain]) if ready else None,
               original_rank_sum=math.fsum(float(v[j]) for j in domain) if ready else None,
               unordered_pair_comparisons=n*(n-1)//2 if ready else 0,nonzero_unordered_pair_comparisons=nonzero)
    return candidate,control,np.array(audit['vol_domain'],bool)&ready,audit


def independent_features(panel,meta,calendar,original,source,limit=None):
    sessions,events,width=schema(panel,meta,calendar)
    require(limit is None or type(limit) is int and 0<=limit<=len(events),'Invalid prefix limit')
    core,cw,calm,v,base,high=inputs(original,source,len(events),width)
    events=events if limit is None else events[:limit]
    values=dict(candidate=np.full((len(events),width),np.nan),control=np.full((len(events),width),np.nan),available=np.zeros((len(events),width),bool))
    history=[];reasons={};total_domain=ready_rows=pairs=nonzero=0
    source_raw=raw_source(panel,meta,sessions,events[-1][0]) if events else None
    for i,(day,names,pos) in enumerate(events):
        require(bool(source_raw['calm'].at[day])==bool(calm[i]) and bool(source_raw['high'].at[day])==bool(high[i]),'Original market states differ')
        raw_vol=source_raw['vol'].loc[day].reindex(names).to_numpy(float)
        raw_core=source_raw['core_calm' if calm[i] else 'core_other'].loc[day].reindex(names).to_numpy(float)
        expected_core,expected_v=ranks(raw_core),ranks(raw_vol)
        require(np.array_equal(expected_v,v[i]),'Original volatility rank parity failed')
        close(core[i]/cw[i],expected_core,'Original core rank',1e-12)
        close(core[i],expected_core*cw[i],'Original weighted core',1e-12)
        # Canonical original percentile, avoiding divide-by-weight roundoff.
        # Missing raw core stays outside valid domains; its neutral rank is .5.
        u=expected_core+.5
        candidate,control,available,row=peer_oracle(raw_vol,raw_core,u,v[i])
        values['candidate'][i]=candidate;values['control'][i]=control;values['available'][i]=available
        row=dict(date=str(day.date()),symbols=names,session_position=pos,calm=bool(calm[i]),highvol=bool(high[i]),
                 core_horizon=252 if calm[i] else 294,raw_vol=json_values(raw_vol),raw_core=json_values(raw_core),
                 core_rank=json_values(u),vol_rank=json_values(v[i]),**row)
        total_domain+=row['domain_count'];ready_rows+=int(row['row_ready']);pairs+=row['unordered_pair_comparisons'];nonzero+=row['nonzero_unordered_pair_comparisons']
        reasons[row['reason']]=reasons.get(row['reason'],0)+1;history.append(row)
    audit=dict(events=len(events),symbols_per_event=width,checked_cells=int(values['available'].size),source_vol_finite_cells=total_domain,
               available_cells=int(values['available'].sum()),unavailable_cells=int((~values['available']).sum()),ready_rows=ready_rows,
               fallback_rows=len(events)-ready_rows,unordered_pair_comparisons=pairs,nonzero_unordered_pair_comparisons=nonzero,
               predictive_model_fits=0,bandwidth=H,replacement_fraction=FRACTION,reason_counts=reasons,event_history=history,
               common_availability=True,ranking_eligibility_filtered=False,portfolio_returns_calculated=False)
    return values,audit


def arrays(values):
    require(isinstance(values,dict) and set(values)=={'candidate','control','available'},'Feature keys')
    a={k:np.asarray(v) for k,v in values.items()};mask=a['available']
    require(mask.ndim==2 and mask.dtype==bool,'Boolean matrix support')
    for key in ('candidate','control'):
        require(a[key].shape==mask.shape and np.issubdtype(a[key].dtype,np.floating),'Feature matrix shape/dtype')
        require(not np.isinf(a[key]).any() and np.array_equal(np.isfinite(a[key]),mask),'Common feature availability')
        require(np.all(np.abs(a[key][mask])<=.5+1e-12),'Peer rank bound')
    return a


def verify_features(panel,meta,calendar,original,source,values,audit):
    got=arrays(values);wanted,expected=independent_features(panel,meta,calendar,original,source)
    require(np.array_equal(got['available'],wanted['available']),'Independent feature support')
    for k in ('candidate','control'): close(got[k],wanted[k],'Independent '+k)
    record_check(audit,expected)
    for i,row in enumerate(expected['event_history']):
        if row['identity_branch']:
            for key in ('candidate','control'):
                mask=got['available'][i]
                require(np.array_equal(got[key][i,mask],np.asarray(source['vol'])[i,mask]),'Identity must exactly copy original V')
    count_keys=('events','symbols_per_event','checked_cells','source_vol_finite_cells','available_cells','unavailable_cells',
                'ready_rows','fallback_rows','unordered_pair_comparisons','nonzero_unordered_pair_comparisons')
    return dict(passed=True,**{k:expected[k] for k in count_keys},independent_scalar_rebuild=True,
                original_raw_core_vol_checked=True,original_core_vol_rank_parity_checked=True,original_market_states_checked=True,
                unchanged_volatility_domain_checked=True,symmetric_fixed_n_peer_operator_checked=True,
                mass_matched_uniform_control_checked=True,identity_branches_exact_checked=True,exact_common_availability=True,
                no_future_or_outcome_inputs=True,audit_checked=True,predictive_model_fits=0,computes_returns=False,
                portfolio_returns_calculated=False)


def independent_deltas(values,original,source,limit=None):
    a=arrays(values);total,width=np.asarray(original['core']).shape
    n=len(a['available']) if limit is None else limit
    require(type(n) is int and 0<=n<=len(a['available'])<=total,'Invalid delta prefix')
    core,cw,calm,v,base,high=inputs(original,source,total,width)
    require(a['available'].shape[1]==width,'Delta matrix width')
    weak=vector(original['state_weak_narrow'],total,'weak',True)
    ext=matrix(original['extension_available'],(total,width),'extension',True)
    forecast=matrix(original['forecast_available'],(total,width),'forecast',True)
    eligible=matrix(original['eligible'],(total,width),'eligible',True)
    rank=matrix(original['rank_log_excess'],(total,width),'rank_log_excess')
    scope=np.zeros((n,width),bool);effective=np.empty((n,width))
    result={k:np.zeros((n,width)) for k in ('candidate','control')}
    for i in range(n):
        for j in range(width):
            scope[i,j]=bool(calm[i] and not weak[i] and ext[i,j] and forecast[i,j] and eligible[i,j] and math.isfinite(float(rank[i,j])))
            budget=.025*bool(calm[i])+.05*bool(weak[i])+.01*bool(scope[i,j])
            effective[i,j]=float(base[i,j])*(1.-budget/(1.-float(cw[i,j])))
            require(math.isfinite(effective[i,j]) and 0.<=effective[i,j]<=base[i,j],'Invalid effective volatility coefficient')
            if i and a['available'][i,j]:
                for key in result:
                    result[key][i,j]=FRACTION*effective[i,j]*(float(a[key][i,j])-float(v[i,j]))
                    require(math.isfinite(result[key][i,j]),'Nonfinite replacement delta')
    detail=dict(events=n,width=width,dose=FRACTION,bandwidth=H,initial_delta_zero=True,
                scope_active=scope.tolist(),base_vol_weights=base[:n].tolist(),effective_vol_weights=effective.tolist(),
                changed_cells={k:int(np.count_nonzero(z)) for k,z in result.items()},available_cells=int(a['available'][:n].sum()))
    return result,detail


def verify_deltas(values,original,source,deltas,audit):
    require(isinstance(deltas,dict) and set(deltas)=={'candidate','control'},'Delta keys')
    wanted,expected=independent_deltas(values,original,source)
    a=arrays(values)
    for key in wanted:
        close(deltas[key],wanted[key],'Independent coefficient delta '+key,atol=2e-17)
        actual=np.asarray(deltas[key]);require(np.isfinite(actual).all(),'Nonfinite deltas')
        neutral=~a['available'].copy()
        if len(neutral): neutral[0]=True
        v=np.asarray(source['vol'])[:len(neutral)]
        neutral|=a['available']&(a[key]==v)
        require(np.array_equal(actual[neutral],np.zeros(int(neutral.sum()))),'Neutral or identity delta must be exact zero')
    record_check(audit,expected)
    return dict(passed=True,events=expected['events'],symbols_per_event=expected['width'],
                checked_cells=int(a['available'].size),available_cells=expected['available_cells'],independent_scalar_rebuild=True,
                original_effective_vol_coefficient_checked=True,scope_active_mask_checked=True,half_slot_replacement_checked=True,
                initial_delta_zero=True,unavailable_exact_zero=True,original_support_unchanged=True,
                no_extra_rank_or_model_filter=True,identity_delta_exact_zero=True,audit_checked=True,
                replacement_fraction=FRACTION,predictive_model_fits=0,episode_applications=0,
                changed_cells=expected['changed_cells'],audit=expected)
