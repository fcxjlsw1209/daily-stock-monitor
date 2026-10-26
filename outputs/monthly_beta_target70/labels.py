"""Paired unit-beta and signal-time beta log labels on an explicit common cohort.

Historical beta is computed independently of all future-label masks. The cached
source log target is copied exactly; only the two new targets are recentered.
"""
from pathlib import Path
from collections import Counter
import hashlib
import importlib.util
import math
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
O=P.parent
CALENDAR_LABELS=O/'monthly_horizon70_research/labels.py'
_SPEC=importlib.util.spec_from_file_location('beta_target70_calendar',CALENDAR_LABELS)
calendar_labels=importlib.util.module_from_spec(_SPEC);_SPEC.loader.exec_module(calendar_labels)
REFERENCE_PATHS=(P/'DESIGN.json',O/'monthly_stability_monitor/inputs.py',
                 CALENDAR_LABELS,calendar_labels.CALENDAR_PATH)
SOURCE_TARGETS=('simple_excess','log_excess','vol_scaled','loss_utility')
TARGETS=('log_beta1','log_beta252')
WINDOW=252
PRICES=253
CLIP=1.


def calendar_provenance():
    provenance=calendar_labels._CAL.provenance()
    return dict(provenance=provenance,source_hashes=calendar_labels._source_hashes(provenance))


def log_ratio(a,b):
    a,b=float(a),float(b)
    if not (math.isfinite(a) and math.isfinite(b) and a>0. and b>0.):
        raise ValueError('Log ratio requires positive finite prices')
    difference=a-b
    if abs(difference)<=b/2.:
        value=math.log1p(difference/b)
    else:
        ma,ea=math.frexp(a);mb,eb=math.frexp(b)
        value=math.log(ma/mb)+(ea-eb)*math.log(2.)
    if not math.isfinite(value):raise FloatingPointError('Nonfinite log price ratio')
    return value


def log_returns(prices):
    p=np.asarray(prices,float)
    if p.shape!=(PRICES,):raise ValueError('Exactly253 historical prices required')
    if not (np.isfinite(p)&(p>0.)).all():raise ValueError('Positive finite historical prices required')
    return np.array([log_ratio(a,b) for a,b in zip(p[1:],p[:-1])])


def _center(values):
    v=np.asarray(values,float)
    if v.shape!=(WINDOW,) or not np.isfinite(v).all():
        raise ValueError('Exactly252 finite daily log returns required')
    delta=[float(x)-float(v[0]) for x in v]
    if not all(math.isfinite(x) for x in delta):raise FloatingPointError('Nonfinite centered log return')
    mean=math.fsum(delta)/WINDOW
    result=np.array([x-mean for x in delta])
    if not np.isfinite(result).all():raise FloatingPointError('Nonfinite centered log return')
    return result


def _dot(a,b):
    value=math.fsum(float(x)*float(y) for x,y in zip(a,b))
    if not math.isfinite(value):raise FloatingPointError('Nonfinite beta moment')
    return value


def beta_from_returns(asset,market):
    """Centered OLS slope; no intercept subtraction, shrinkage, cap or epsilon."""
    x,y=_center(market),_center(asset)
    variance=_dot(x,x)
    if variance==0.:return np.nan,'zero_spy_variance'
    beta=_dot(x,y)/variance
    if not math.isfinite(beta):raise FloatingPointError('Nonfinite historical beta')
    return beta,'available'


def raw_log_targets(asset_log,market_log,beta):
    if not all(math.isfinite(v) for v in (asset_log,market_log,beta)):
        raise FloatingPointError('Nonfinite target component')
    raw=(asset_log-market_log,asset_log-beta*market_log)
    if not all(math.isfinite(v) for v in raw):raise FloatingPointError('Nonfinite raw beta target before clipping')
    return dict(zip(TARGETS,raw))


def _days(values,name):
    v=pd.DatetimeIndex(values)
    if v.hasnans or v.tz is not None or not v.equals(v.normalize()):
        raise ValueError(name+' requires naive midnight dates')
    return v.to_numpy(dtype='datetime64[D]')


def _bool(values,n,name):
    v=np.asarray(values)
    if v.shape!=(n,) or v.dtype.kind!='b':raise ValueError(name+' requires an exact boolean row vector')
    return v


def query_plan(day,calendar):
    day=pd.Timestamp(day)
    path=pd.DatetimeIndex(calendar.sessions_in_range(day-pd.DateOffset(years=2),day))
    if path.tz is not None:path=path.tz_localize(None)
    if (path.hasnans or not path.is_unique or not path.is_monotonic_increasing
            or not path.equals(path.normalize())):raise ValueError('Invalid official session sequence')
    if not len(path) or path[-1]!=day:raise ValueError('Signal must be an official session')
    if len(path)<PRICES:raise ValueError('Insufficient official calendar history')
    return path[-PRICES:]


def _prices(adj,path,symbol):
    if symbol not in adj.columns or not path.isin(adj.index).all():return None
    try:p=adj.loc[path,symbol].to_numpy(float,copy=True,na_value=np.nan)
    except (TypeError,ValueError):return None
    return p if (np.isfinite(p)&(p>0.)).all() else None


def build_targets(adj,dates,ends,symbols,train_plain,source_targets,common_good,*,calendar=None):
    """Return paired labels, masks, independent betas and untouched source label.

    targets/raw_targets have log_beta1/log_beta252 keys. paired_good is a subset
    of source_good, which matches the old four-target/ten-feature finite cohort.
    Both paired labels clip before centering over the exact same month rows.
    beta/availability are computed for every historical row, even source-excluded
    rows, without reading their label ends. Endpoints are only read for labels.
    """
    if not isinstance(adj,pd.DataFrame) or not isinstance(adj.index,pd.DatetimeIndex):
        raise TypeError('Adjusted price DataFrame required')
    if (adj.index.hasnans or adj.index.tz is not None or not adj.index.equals(adj.index.normalize())
            or not adj.index.is_unique or not adj.index.is_monotonic_increasing or not adj.columns.is_unique):
        raise ValueError('Unique increasing naive midnight price axes required')
    dates,ends=_days(dates,'dates'),_days(ends,'ends');n=len(dates)
    if ends.shape!=(n,) or (ends<=dates).any() or np.any(ends.astype('datetime64[M]')!=dates.astype('datetime64[M]')+1):
        raise ValueError('Labels must end strictly after signals in the next calendar month')
    symbols=np.asarray(symbols)
    if symbols.shape!=(n,) or any(not isinstance(s,str) or not s or s!=s.strip() for s in symbols):
        raise ValueError('Explicit aligned security identifiers required')
    X=np.asarray(train_plain,float)
    if X.shape!=(n,10):raise ValueError('Original plain10 matrix required')
    common=_bool(common_good,n,'common_good')
    if set(source_targets)!=set(SOURCE_TARGETS):raise ValueError('All four original targets required')
    original={key:np.asarray(source_targets[key],float) for key in SOURCE_TARGETS}
    if any(v.shape!=(n,) for v in original.values()):raise ValueError('Original target lengths differ')
    source_log_target=original['log_excess'].copy()
    source_good=common & np.isfinite(X).all(axis=1)
    for value in original.values():source_good &= np.isfinite(value)
    months=dates.astype('datetime64[M]')
    groups=[np.flatnonzero(months==month) for month in np.unique(months)]
    for rows in groups:
        if len(np.unique(dates[rows]))!=1 or len(np.unique(ends[rows]))!=1 or len(set(symbols[rows]))!=len(rows):
            raise ValueError('Each month needs one signal/end and unique securities')
        if source_good[rows].sum()<2:source_good[rows]=False
    if calendar is None:
        left=pd.Timestamp(dates.min() if n else '2002-01-01')-pd.DateOffset(years=2,months=1)
        right=pd.Timestamp(dates.max() if n else '2002-01-01')
        calendar=calendar_labels._CAL.market_calendar(start=str(left.date()),end=str(right.date()))
    beta=np.full(n,np.nan);beta_available=np.zeros(n,bool);endpoint_available=np.zeros(n,bool)
    targets={key:np.full(n,np.nan) for key in TARGETS};raw={key:np.full(n,np.nan) for key in TARGETS}
    paired=np.zeros(n,bool);records=[];reasons=Counter()
    for rows in groups:
        day,end=pd.Timestamp(dates[rows[0]]),pd.Timestamp(ends[rows[0]])
        path=query_plan(day,calendar);missing=path[~path.isin(adj.index)]
        spy=_prices(adj,path,'SPY') if not len(missing) else None
        market=log_returns(spy) if spy is not None else None
        market_reason='missing_official_session' if len(missing) else 'invalid_spy_path'
        if market is not None:
            market_reason='zero_spy_variance' if _dot(_center(market),_center(market))==0. else 'available'
        statuses={}
        # Historical betas do not consult source_good, target endpoints or labels.
        for row in rows:
            reason=market_reason
            if reason=='available':
                p=_prices(adj,path,symbols[row])
                if p is None:reason='invalid_asset_path'
                else:
                    value,reason=beta_from_returns(log_returns(p),market)
                    if reason=='available':beta[row]=value;beta_available[row]=True
            statuses[str(symbols[row])]=reason;reasons[reason]+=1
        endpoint_path=pd.DatetimeIndex([day,end]);spy_end=_prices(adj,endpoint_path,'SPY')
        market_log=log_ratio(spy_end[1],spy_end[0]) if spy_end is not None else None
        for row in rows[source_good[rows]&beta_available[rows]]:
            p=_prices(adj,endpoint_path,symbols[row])
            if p is None or market_log is None:continue
            values=raw_log_targets(log_ratio(p[1],p[0]),market_log,float(beta[row]))
            endpoint_available[row]=True
            for key in TARGETS:raw[key][row]=values[key]
        selected=rows[source_good[rows]&beta_available[rows]&endpoint_available[rows]]
        counts={key:0 for key in TARGETS}
        if len(selected)>=2:
            paired[selected]=True
            for key in TARGETS:
                values=raw[key][selected];counts[key]=int((np.abs(values)>CLIP).sum())
                clipped=np.clip(values,-CLIP,CLIP)
                targets[key][selected]=clipped-clipped.mean()
        records.append(dict(signal_date=str(day.date()),label_end=str(end.date()),
            first_beta_price=str(path[0].date()),last_beta_price=str(path[-1].date()),
            beta_price_count=PRICES,beta_return_count=WINDOW,rows=int(len(rows)),
            missing_official_sessions=[str(d.date()) for d in missing],beta_status=statuses,
            known_beta_rows=int(beta_available[rows].sum()),source_rows=int(source_good[rows].sum()),
            source_missing_beta_rows=int((source_good[rows]&~beta_available[rows]).sum()),
            source_known_beta_missing_endpoint_rows=int((source_good[rows]&beta_available[rows]&~endpoint_available[rows]).sum()),
            paired_before_minimum_rows=int(len(selected)),paired_rows=int(paired[rows].sum()),
            singleton_rows_removed=int(len(selected)) if len(selected)<2 else 0,clip_counts=counts))
    provenance=calendar_provenance();paths=(Path(__file__).resolve(),)+REFERENCE_PATHS
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};hashes.update(provenance['source_hashes'])
    audit=dict(beta_window_returns=WINDOW,beta_prices=PRICES,beta_clock='official253 prices ending at historical signal close',
        beta_formula='dot(centered_SPY_logreturns,centered_asset_logreturns)/dot(centered_SPY_logreturns,centered_SPY_logreturns)',
        beta_uses_future_labels=False,beta_uses_source_good=False,negative_beta_valid=True,flat_asset_beta_zero_valid=True,
        market_variance_zero='unknown; no epsilon',intercept_subtracted=False,beta_capped_or_shrunk=False,
        targets=list(TARGETS),target_clip_before_month_center=CLIP,minimum_month_rows=2,
        source_log_target='exact copy of cached source_targets.log_excess; not rebuilt or recentered',
        source_rows=int(source_good.sum()),paired_rows=int(paired.sum()),removed_source_rows=int((source_good&~paired).sum()),
        all_four_original_targets_and_ten_features_validate_source=True,labels_share_exact_cohort=True,
        original_future_path_rule='retained through common_good',no_query_eligibility_change=True,
        beta_reasons=dict(reasons),months=records,calendar=provenance,source_hashes=hashes,
        dependencies=dict(numpy=np.__version__,pandas=pd.__version__))
    return dict(targets=targets,raw_targets=raw,paired_good=paired,source_good=source_good,
        source_log_target=source_log_target,beta=beta,beta_available=beta_available,audit=audit)
