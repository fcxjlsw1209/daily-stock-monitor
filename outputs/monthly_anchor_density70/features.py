"""Causal anchor union and exact original plain10 reconstruction; no labels."""
from pathlib import Path
import hashlib
import importlib
import importlib.util
import sys
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
O = P.parent
ARCH_PATH = O/'monthly_architecture_restart/study.py'
CALENDAR_PATH = O/'monthly_earnings_repair/research_calendar.py'
LEGACY_PATH = O/'monthly_reconstruction_v2/foreign/membership.csv.gz'
_saved_path = list(sys.path)
try:
    sys.path.insert(0, str(ARCH_PATH.parent))
    _sp = importlib.util.spec_from_file_location('density70_arch', ARCH_PATH)
    arch = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(arch)
finally:
    sys.path[:] = _saved_path
_sp = importlib.util.spec_from_file_location('density70_calendar', CALENDAR_PATH)
calendar_source = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(calendar_source)


def dependency_files():
    provenance = calendar_source.provenance()
    paths = {Path(__file__).resolve(), ARCH_PATH, CALENDAR_PATH,
             O/'monthly_state_rank_research/state_rank.py',
             O/'monthly_stability_monitor/inputs.py', Path(provenance['manifest']),
             Path(arch.ex.t.strict.__file__).resolve()}
    paths.update(Path(row['source_file']) for row in provenance['closures'])
    for name in ('exchange_calendars.exchange_calendar_xnys', 'exchange_calendars.exchange_calendar',
                 'exchange_calendars.us_holidays', 'exchange_calendars.calendar_helpers', 'pandas.tseries.holiday'):
        paths.add(Path(importlib.import_module(name).__file__).resolve())
    return tuple(sorted(paths))


def calendar_provenance():
    return dict(provenance=calendar_source.provenance(), source_hashes={
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependency_files()})


def dates(values, name):
    d = pd.DatetimeIndex(values)
    if (d.empty or d.hasnans or d.tz is not None or not d.equals(d.normalize())
            or not d.is_unique or not d.is_monotonic_increasing):
        raise ValueError(name+' must be nonempty, increasing unique naive session dates')
    return d


def pools(values, n):
    p = [list(ss) for ss in values]
    if len(p) != n or any(not ss or len(set(ss)) != len(ss) or
                         any(not isinstance(s, str) or not s for s in ss) for ss in p):
        raise ValueError('Explicit unique nonempty symbol pools required')
    return p


def sessions(calendar, left, right):
    s = pd.DatetimeIndex(calendar.sessions_in_range(pd.Timestamp(left), pd.Timestamp(right)))
    if s.tz is not None: s = s.tz_localize(None)
    if (s.hasnans or not s.is_unique or not s.is_monotonic_increasing or
            not s.equals(s.normalize()) or (s < pd.Timestamp(left)).any() or
            (s > pd.Timestamp(right)).any()):
        raise ValueError('Invalid official session list')
    return s


def calendar_for(ds):
    d = pd.DatetimeIndex(ds)
    return calendar_source.market_calendar(
        start=str((d.min().to_period('M')-1).start_time.date()),
        end=str((d.max().to_period('M')+3).end_time.date()))


def _membership(frame, name):
    if not isinstance(frame, pd.DataFrame): raise ValueError(name+' must be a frame')
    dates(frame.index, name+' dates')
    if not frame.columns.is_unique or frame.dtypes.ne(bool).any():
        raise ValueError(name+' must retain unique symbols and exact boolean membership')


def anchor_plan(train_meta, member, legacy_member, etfs, calendar=None):
    """Monthly snapshots retain exact order; only NEW weekly anchors use past pools."""
    td = dates(train_meta['dates'], 'monthly dates')
    ts = pools(train_meta['symbols'], len(td))
    if len(set(td.to_period('M'))) != len(td): raise ValueError('One original anchor per month required')
    _membership(member, 'archive'); _membership(legacy_member, 'legacy')
    etfs = set(etfs)
    if any(not isinstance(s, str) or not s for s in etfs): raise ValueError('Explicit ETF identifiers required')
    calendar = calendar or calendar_for(td)
    left, right = td[0].to_period('M').start_time, td[-1].to_period('M').end_time.normalize()
    # Expand before grouping: never mislabel a truncated week as its last session.
    full = sessions(calendar, left-pd.Timedelta(days=7), right+pd.Timedelta(days=7))
    weekly = full.to_series().groupby(full.to_period('W-FRI')).last()
    weekly = set(weekly[(weekly >= left) & (weekly <= right)])
    for d in td:
        sm = sessions(calendar, d.to_period('M').start_time, d.to_period('M').end_time.normalize())
        if not len(sm) or d != sm[-1]: raise ValueError('Original monthly anchor differs from official month end')
    monthly = dict(zip(td, ts)); union = sorted(set(td) | weekly)
    original_origins = train_meta.get('pool_origins')
    if original_origins is not None and len(original_origins) != len(td):
        raise ValueError('Original monthly pool origins must align')
    original_origins = dict(zip(td, original_origins)) if original_origins is not None else {}
    out = dict(dates=[], symbols=[], pool_origins=[], is_monthly=[], offsets=[0])
    for day in union:
        prior = member.index[member.index <= day]
        if day in monthly:
            ss = monthly[day]
            origin = original_origins.get(day, str(prior[-1].date()) if len(prior) else
                                          'inherited_pre2016_historical_proxy')
            if origin != 'inherited_pre2016_historical_proxy' and pd.Timestamp(origin) > day:
                raise ValueError('Original monthly pool origin is in the future')
        elif len(prior):
            origin = str(prior[-1].date())
            ss = sorted(set(member.columns[member.loc[prior[-1]]]) | etfs)
        else:
            if day not in legacy_member.index: raise ValueError('Missing historical proxy date '+str(day.date()))
            origin = 'inherited_pre2016_historical_proxy'
            ss = sorted(set(legacy_member.columns[legacy_member.loc[day]]) | etfs)
        if not ss: raise ValueError('Empty anchor pool')
        out['dates'].append(str(day.date())); out['symbols'].append(ss)
        out['pool_origins'].append(origin); out['is_monthly'].append(day in monthly)
        out['offsets'].append(out['offsets'][-1]+len(ss))
    return out


def snapshots(panel, ds, ss, etfs):
    """Use unchanged source row clock/rank/neutral handling, not old weekly84."""
    ds = dates(ds, 'snapshot dates'); ss = pools(ss, len(ds))
    adj = panel['adj']
    dates(adj.index, 'panel dates')
    if not adj.columns.is_unique: raise ValueError('Duplicate panel security')
    last = ds[-1]
    cut = {k: v.loc[:last] for k, v in panel.items() if k != 'member'}
    f, mom, good = arch.raw_frames(cut)
    xs, gs = [], []
    for day, names in zip(ds, ss):
        if day not in adj.index or any(s not in adj for s in names):
            raise ValueError('Source daily panel lacks explicit snapshot axes')
        row, g, aux = arch.snapshot(day, names, f, mom, good, cut, set(etfs), 'all')
        xs.append(np.column_stack([row[k] for k in ['raw', 'recent_risk', 'vol', 'sector']]+[aux[:, j] for j in range(6)]))
        gs.append(g)
    return np.vstack(xs), np.concatenate(gs)


def build_inputs(panel, train_meta, event_meta, train_plain, event_plain,
                 train_eligible, event_eligible, *, legacy_member, calendar=None):
    etfs = event_meta['etfs']
    plan = anchor_plan(train_meta, panel['member'], legacy_member, etfs, calendar)
    X, g = snapshots(panel, plan['dates'], plan['symbols'], etfs)
    Q, qg = snapshots(panel, event_meta['dates'], event_meta['symbols'], etfs)
    widths = np.diff(plan['offsets'])
    d = np.repeat(np.array(plan['dates'], dtype='datetime64[D]'), widths)
    monthly = np.repeat(np.array(plan['is_monthly'], dtype=bool), widths)
    idx = np.flatnonzero(monthly)
    source_X, source_Q = np.asarray(train_plain), np.asarray(event_plain)
    tg, eg = np.asarray(train_eligible), np.asarray(event_eligible).reshape(-1)
    if source_X.dtype != X.dtype or source_Q.dtype != Q.dtype:
        raise ValueError('Source plain10 dtype differs')
    if tg.dtype.kind != 'b' or eg.dtype.kind != 'b': raise ValueError('Exact source eligibility booleans required')
    np.testing.assert_array_equal(X[idx], source_X, err_msg='Original monthly plain10 reconstruction differs')
    np.testing.assert_array_equal(Q, source_Q, err_msg='Original query plain10 reconstruction differs')
    np.testing.assert_array_equal(g[idx], tg, err_msg='Original monthly eligibility differs')
    np.testing.assert_array_equal(qg, eg, err_msg='Original query eligibility differs')
    arrays = dict(density_X=X, density_eligible=g, density_dates=d,
                  density_symbols=np.array(sum(plan['symbols'], []), dtype=str),
                  density_is_monthly=monthly, monthly_row_index=idx)
    return dict(arrays=arrays, meta=plan, audit=dict(monthly_anchors=len(train_meta['dates']),
        union_anchors=len(plan['dates']), new_weekly_anchors=sum(not x for x in plan['is_monthly']),
        monthly_query_plain10_and_eligibility_exact=True, source_row_clock=True,
        future_month_pool_used=False, legacy_proxy_not_certified_PIT=True,
        weekly_origin_rule='latest archived date <= signal, else inherited daily legacy proxy'))
