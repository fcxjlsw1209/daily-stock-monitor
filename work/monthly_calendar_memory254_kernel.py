"""Paired monthly episode memory; caller-supplied sources, no research IO.

Historical anchors are official month ends with the original positive signal
and feature-good mask. Both labels share sample validity and the later actual
exit as their information clock. Source and old-incumbent gates precede new
effect estimation. The annual Local67 pre-score is never refitted or changed.
"""
from copy import deepcopy
import hashlib
import importlib.util
from numbers import Real
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD_PATH = ROOT/'work/monthly_calendar_clock248_v2_kernel.py'
OLD_SHA = 'b580b2a2389201f48d81cd78ddecb641699fdbd919c79202746665789b5a6bcb'
EP_PATH = ROOT/'outputs/monthly_daily_episodes/episodes.py'
EP_SHA = '8ae91e55e0d662cb6dee9cba6d1422aa3713b779976de69b4560194d311ae16a'
SIGNAL_PATH = ROOT/'outputs/monthly_company_context/context_research.py'
SIGNAL_SHA = '271fdb223f66a0d890d015073cb2b49664f94eaab9a86ca11742dad29e835f7c'
for path, digest in ((OLD_PATH, OLD_SHA), (EP_PATH, EP_SHA), (SIGNAL_PATH, SIGNAL_SHA)):
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('Frozen dependency changed: '+path.name)

def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

old = _module('calendar254_cache_bridge', OLD_PATH)
ep = _module('calendar254_original_episodes', EP_PATH)
require = old.require
WIDTH = 49
INITIAL_PARTIAL = '2016-08-17'
ARMS = ('incumbent_local67', 'calendar_memory_local67', 'fixed21_memory_control')
POLICY = dict(budget=.05, ceiling=5, gap=.02)
MODEL21 = dict(horizon=21, match='signal', target='utility', mode='level')
SOURCE_SCHEMA = 'monthly_calendar_memory254_features_v1'
SCHEMA = 'monthly_calendar_memory254_decisions_v1'
EVENT_DATA_KEYS = ('event_symbol', 'event_dates', 'event_anchor_index', 'event_signal',
    'event_entry', 'event_exit21', 'event_exit_calendar', 'event_common_maturity',
    'event_valid21', 'event_valid_calendar', 'event_common_valid', 'event_excess21',
    'event_excess_calendar', 'event_utility21', 'event_utility_calendar')
QUERY_DATA_KEYS = ('query_dates', 'query_symbols', 'query_shape', 'query_market',
                   'query_signal', 'query_vol')
DATA_KEYS = EVENT_DATA_KEYS + QUERY_DATA_KEYS
SOURCE_COUNT_FIELDS = ('queries', 'symbols_per_query', 'securities', 'anchors',
    'valid21_anchors', 'valid_calendar_anchors', 'common_valid_anchors',
    'common_invalid_anchors', 'initial_partial_queries', 'formal_queries',
    'query_cells', 'finite_query_cells', 'positive_signal_query_cells',
    'new_model_fits', 'predictive_model_fits')
COUNT_FIELDS = ('events', 'symbols_per_event', 'initial_partial_events', 'formal_events',
    'original_eligible_cells', 'query_signal_cells', 'query_finite_cells',
    'calendar_nonzero_effect_cells', 'fixed21_nonzero_effect_cells',
    'calendar_evidence_cells', 'fixed21_evidence_cells',
    'candidate_changed_holdings_events', 'control_changed_holdings_events',
    'candidate_vs_control_changed_holdings_events', 'original_episode_calls',
    'candidate_episode_calls', 'control_episode_calls', 'anchors',
    'mature_common_anchors', 'new_model_fits', 'predictive_model_fits')
AUDIT_EVENT_KEYS = ('query_index', 'date', 'initial_partial', 'original_eligible_indices',
    'pre_order', 'boundary_candidates', 'episode_budget', 'incumbent_choices',
    'candidate_choices', 'control_choices')
LABEL_KEYS = ('event_excess21', 'event_excess_calendar', 'event_utility21',
              'event_utility_calendar')
DATE_KEYS = ('event_dates', 'event_entry', 'event_exit21', 'event_exit_calendar',
             'event_common_maturity', 'query_dates')
INT_KEYS = ('event_anchor_index', 'event_signal', 'query_signal')
BOOL_KEYS = ('event_valid21', 'event_valid_calendar', 'event_common_valid')


def dependency_files():
    return tuple(dict.fromkeys((OLD_PATH, EP_PATH, SIGNAL_PATH) + old.dependency_files()))


def _meta(meta, n):
    require(type(meta) is dict and isinstance(meta.get('dates'), list) and
            isinstance(meta.get('symbols'), list), 'Metadata axes')
    total = len(meta['dates'])
    n = total if n is None else n
    require(type(n) is int and 0 <= n <= total and len(meta['symbols']) == total,
            'Metadata prefix')
    previous = None
    for i in range(n):
        d = old._iso(meta['dates'][i], 'Query date'); names = meta['symbols'][i]
        require(previous is None or d > previous, 'Increasing queries'); previous = d
        require(d != INITIAL_PARTIAL or i == 0, 'Named partial first')
        require(type(names) is list and len(names) == WIDTH and len(set(names)) == WIDTH and
                all(type(s) is str and s and len(s) <= 64 and s.strip() == s for s in names),
                'Original49 identities')
    return n


def _empty(n=0):
    out = {}
    for key in DATA_KEYS:
        if key in DATE_KEYS: dtype = 'datetime64[D]'
        elif key in INT_KEYS: dtype = 'int64'
        elif key in BOOL_KEYS: dtype = bool
        elif key in ('event_symbol', 'query_symbols'): dtype = 'U64'
        else: dtype = 'float64'
        shape = ((n, WIDTH, 8) if key == 'query_shape' else
                 (n, 3) if key == 'query_market' else
                 (n, WIDTH) if key in ('query_symbols', 'query_signal', 'query_vol') else
                 (n,) if key == 'query_dates' else (0,))
        out[key] = np.empty(shape, dtype=dtype)
    return out


def _calendar(calendar):
    if isinstance(calendar, dict): calendar = calendar.get('sessions')
    return old._calendar(calendar)


def _panel(panel, names, last, sessions):
    fields = ('adj', 'open', 'close', 'high', 'low', 'volume')
    require(type(panel) is dict and all(k in panel for k in fields), 'Six source frames')
    base = panel['adj']
    require(isinstance(base, pd.DataFrame) and isinstance(base.index, pd.DatetimeIndex),
            'Adjusted source frame')
    ix = base.index
    require(ix.tz is None and ix.is_unique and ix.is_monotonic_increasing and
            not ix.hasnans and all(d == d.normalize() for d in ix), 'Source date axis')
    require(base.columns.is_unique, 'Source unique columns')
    needed = sorted(set(names) | {'SPY', 'QQQ'})
    require(set(needed) <= set(base.columns), 'Missing source security/benchmark column')
    idx = ix[ix <= pd.Timestamp(last)]
    require(len(idx) and set(idx.strftime('%Y-%m-%d')) <= set(sessions),
            'Source dates must be official sessions')
    out = {}
    for key in fields:
        frame = panel[key]
        require(isinstance(frame, pd.DataFrame) and frame.index.equals(ix) and
                frame.columns.equals(base.columns), 'Six frame axes agree')
        frame = frame.loc[idx, needed].copy()
        for symbol in needed:
            vals = []
            for value in frame[symbol]:
                if value is None or value is pd.NA:
                    vals.append(np.nan); continue
                require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_)),
                        'Source numeric nonboolean '+key+'/'+symbol)
                try: vals.append(float(value))
                except (OverflowError, ValueError, TypeError) as exc:
                    raise ValueError('Source binary64 '+key+'/'+symbol) from exc
            frame[symbol] = np.asarray(vals, dtype='float64')
        # Missing official rows remain explicit NaNs: no filling and no
        # compression of rolling/shift clocks in the original feature formula.
        grid = pd.DatetimeIndex([d for d in sessions if str(idx[0].date()) <= d <= last])
        out[key] = frame.reindex(grid)
    return out


def _plan(t, sessions, month_last, lookup):
    day = sessions[t]
    require(t+21 < len(sessions) and t+1 < len(sessions), 'Calendar future21 coverage')
    month = pd.Timestamp(day).to_period('M') + 1
    end = month_last.get(str(month))
    require(end is not None, 'Calendar full following month required')
    e = lookup[end]
    require(e+1 < len(sessions) and sessions[e+1][:7] != str(month),
            'Calendar following month witness')
    return sessions[t+1], sessions[t+21], end, max(sessions[t+21], end)


def _label(panel, opening, symbol, t, end, sessions):
    dates = pd.DatetimeIndex(sessions[t: end+1])
    px = panel['adj']
    if not dates.isin(px.index).all(): return np.nan, np.nan, False
    # Only required entry and closes are read by the frozen mathematical helper.
    asset = px.loc[dates, symbol].to_numpy(dtype=float)
    market = px.loc[dates, 'SPY'].to_numpy(dtype=float)
    op = opening.loc[dates, symbol].to_numpy(dtype=float)
    mo = opening.loc[dates, 'SPY'].to_numpy(dtype=float)
    with np.errstate(all='ignore'):
        excess, utility = ep.label_path(asset, op, market, mo, 0, end-t)
    valid = bool(np.isfinite(excess) and np.isfinite(utility))
    return (float(excess), float(utility), True) if valid else (np.nan, np.nan, False)


def _finite_queries(data):
    return (np.isfinite(data['query_shape']).all(axis=2) &
            np.isfinite(data['query_market']).all(axis=1)[:, None] &
            np.isfinite(data['query_vol']))


def build_features(panel, calendar, meta, n=None):
    n = _meta(meta, n); data = _empty(n)
    if n:
        sessions = _calendar(calendar); lookup = {d: i for i, d in enumerate(sessions)}
        month_last = {d[:7]: d for d in sessions}
        names = sorted(set(s for row in meta['symbols'][:n] for s in row))
        last = meta['dates'][n-1]; source = _panel(panel, names, last, sessions)
        for day in meta['dates'][:n]:
            require(day in lookup and pd.Timestamp(day) in panel['adj'].index,
                    'Missing query source/official session')
            t = lookup[day]
            require(day == INITIAL_PARTIAL or
                    (t+1 < len(sessions) and sessions[t+1][:7] != day[:7]),
                    'Ordinary query must be official monthend')
        with np.errstate(all='ignore'):
            shape, market, signal, good, vol = ep.features(source)
            opening = source['adj'] * source['open'] / source['close']
        rows = {key: [] for key in EVENT_DATA_KEYS}
        monthends = [i for i, day in enumerate(sessions[:-1]) if day <= last and
            sessions[i+1][:7] != day[:7] and pd.Timestamp(day) in source['adj'].index]
        for symbol in names:
            for t in monthends:
                day = sessions[t]
                if not (bool(good.at[pd.Timestamp(day), symbol]) and
                        int(signal.at[pd.Timestamp(day), symbol]) > 0): continue
                entry, exit21, exitcal, maturity = _plan(t, sessions, month_last, lookup)
                ex21, u21, v21 = _label(source, opening, symbol, t, t+21, sessions)
                excal, ucal, vc = _label(source, opening, symbol, t, lookup[exitcal], sessions)
                values = (symbol, day, t, int(signal.at[pd.Timestamp(day), symbol]),
                    entry, exit21, exitcal, maturity, v21, vc, v21 and vc,
                    ex21, excal, u21, ucal)
                for key, value in zip(EVENT_DATA_KEYS, values): rows[key].append(value)
        for key in EVENT_DATA_KEYS: data[key] = np.asarray(rows[key], dtype=data[key].dtype)
        for i, day in enumerate(meta['dates'][:n]):
            ss = meta['symbols'][i]
            data['query_dates'][i] = day; data['query_symbols'][i] = ss
            data['query_shape'][i] = np.column_stack([f.loc[day, ss].to_numpy() for f in shape])
            data['query_market'][i] = market.loc[day].to_numpy()
            data['query_signal'][i] = signal.loc[day, ss].to_numpy(dtype=np.int64)
            data['query_vol'][i] = vol.loc[day, ss].to_numpy()
    else: names = []; last = None
    counts = dict(queries=n, symbols_per_query=WIDTH, securities=len(names),
        anchors=len(data['event_dates']), valid21_anchors=int(data['event_valid21'].sum()),
        valid_calendar_anchors=int(data['event_valid_calendar'].sum()),
        common_valid_anchors=int(data['event_common_valid'].sum()),
        common_invalid_anchors=int((~data['event_common_valid']).sum()),
        initial_partial_queries=sum(d == INITIAL_PARTIAL for d in meta['dates'][:n]),
        formal_queries=sum(d != INITIAL_PARTIAL for d in meta['dates'][:n]), query_cells=n*WIDTH,
        finite_query_cells=int(_finite_queries(data).sum()),
        positive_signal_query_cells=int((data['query_signal'] > 0).sum()),
        new_model_fits=0, predictive_model_fits=0)
    require(tuple(counts) == SOURCE_COUNT_FIELDS, 'Source count schema')
    audit = dict(schema=SOURCE_SCHEMA, counts=counts,
        anchor_rule='official_monthends_original_good_positive_signal_no_lockout',
        common_maturity_rule='maximum_of_actual_exits', numeric_cutoff=last,
        candidate_selection_computed=False, portfolio_returns_computed=False)
    return data, audit


def _layout(data):
    require(type(data) is dict and set(data) == set(DATA_KEYS), 'Episode data exact keys')
    require(all(isinstance(data[k], np.ndarray) for k in DATA_KEYS), 'Episode arrays required')
    total = len(data['query_dates']); events = len(data['event_dates'])
    for key in DATA_KEYS:
        if key in DATE_KEYS: dtype = np.dtype('datetime64[D]')
        elif key in INT_KEYS: dtype = np.dtype('int64')
        elif key in BOOL_KEYS: dtype = np.dtype(bool)
        elif key in ('event_symbol', 'query_symbols'): dtype = np.dtype('U64')
        else: dtype = np.dtype('float64')
        shape = ((total, WIDTH, 8) if key == 'query_shape' else
                 (total, 3) if key == 'query_market' else
                 (total, WIDTH) if key in ('query_symbols', 'query_signal', 'query_vol') else
                 (total,) if key == 'query_dates' else (events,))
        require(data[key].shape == shape and data[key].dtype == dtype,
                'Episode dtype/shape '+key)
    return total


def data_prefix(data, n):
    total = _layout(data)
    require(type(n) is int and 0 <= n <= total, 'Episode prefix')
    if n:
        names = np.unique(data['query_symbols'][:n])
        mask = ((data['event_dates'] <= data['query_dates'][n-1]) &
                np.isin(data['event_symbol'], names))
    else: mask = np.zeros(len(data['event_dates']), dtype=bool)
    return {key: (data[key][mask].copy() if key in EVENT_DATA_KEYS else
                  data[key][:n].copy()) for key in DATA_KEYS}


def _validate(data, meta, n):
    out = data_prefix(data, n)
    require(np.array_equal(out['query_dates'], np.asarray(meta['dates'][:n], dtype='datetime64[D]')) and
            np.array_equal(out['query_symbols'], np.asarray(meta['symbols'][:n], dtype='U64').reshape(n, WIDTH)),
            'Query identity parity')
    require(not np.isnat(out['query_dates']).any(), 'Query dates valid')
    require(((out['query_signal'] >= -1) & (out['query_signal'] <= 3)).all(), 'Query signal types')
    for key in DATE_KEYS[:-1]: require(not np.isnat(out[key]).any(), 'Event dates valid')
    identities = list(zip(out['event_symbol'].tolist(), out['event_dates'].astype(str).tolist()))
    require(identities == sorted(set(identities)), 'Unique symbol/date ordered anchors')
    require((out['event_anchor_index'] >= 0).all() and
            ((out['event_signal'] > 0) & (out['event_signal'] <= 3)).all(), 'Anchor metadata')
    require((out['event_entry'] > out['event_dates']).all() and
            (out['event_exit21'] >= out['event_entry']).all() and
            (out['event_exit_calendar'] >= out['event_entry']).all() and
            np.array_equal(out['event_common_maturity'],
                np.maximum(out['event_exit21'], out['event_exit_calendar'])), 'Actual exits/common maturity')
    if n:
        mature = out['event_common_maturity'] <= out['query_dates'][-1]
        require(np.array_equal(out['event_common_valid'][mature],
            (out['event_valid21'] & out['event_valid_calendar'])[mature]), 'Paired common mask')
        for suffix, valid in (('21', 'event_valid21'), ('_calendar', 'event_valid_calendar')):
            for prefix in ('event_excess', 'event_utility'):
                value = out[prefix+suffix]; used = mature & out[valid]
                require(np.isfinite(value[used]).all() and (np.abs(value[used]) <= 1.).all(), 'Valid label bounded')
                require(np.isnan(value[mature & ~out[valid]]).all(), 'Unavailable label NaN')
    return out


def _forecast(data, good, n):
    shape = (n, WIDTH); result = {}; enough = {}; finite = _finite_queries(data)
    for arm, label in ((ARMS[1], 'event_utility_calendar'), (ARMS[2], 'event_utility21')):
        effect = np.zeros(shape, dtype=float); evidence = np.zeros(shape, dtype=bool)
        # Array masking is causal at each query inside frozen estimate().
        labels = data[label].copy(); labels[~data['event_common_valid']] = np.nan
        events = dict(symbol=data['event_symbol'], dates=data['event_dates'],
            ends_21=data['event_common_maturity'], signal=data['event_signal'], utility_21=labels)
        for i in range(n):
            if str(data['query_dates'][i]) == INITIAL_PARTIAL: continue
            for j in range(WIDTH):
                query = dict(symbol=str(data['query_symbols'][i,j]), date=data['query_dates'][i],
                    eligible=bool(good[i,j] and finite[i,j]), shape=data['query_shape'][i,j],
                    market=data['query_market'][i], signal=int(data['query_signal'][i,j]),
                    vol=float(data['query_vol'][i,j]))
                value, info = ep.estimate(events, query, MODEL21)
                require(np.isfinite(value) and abs(value) <= .5, 'Bounded episode effect')
                effect[i,j] = value; evidence[i,j] = info['effective_samples'] >= 3
        result[arm] = effect; enough[arm] = evidence
    return result, enough


def decisions(study, arrays, meta, data, n=None, *, source_validator=None,
              incumbent_validator=None, expected_incumbent_orders=None):
    n = _meta(meta, n); before = old._snapshot(arrays)
    metadata = {k: deepcopy(meta[k][:n]) for k in ('dates', 'ends', 'symbols')}
    metadata['etfs'] = deepcopy(meta.get('etfs'))
    view = _validate(data, meta, n)
    if source_validator is not None:
        require(callable(source_validator), 'Source validator callable')
        receipt = source_validator({k: v.copy() for k,v in view.items()})
        require(type(receipt) is dict and receipt.get('passed') is True, 'Independent source gate failed')
    b = old._cache(study, arrays, meta, n)
    require(b['width'] == WIDTH, 'Original49 cache')
    source = dict(baseline=b['base'], core=b['core'], core_weight=b['cw'],
        eligible=b['good'], state_calm10=b['calm'], state_weak_narrow=b['weak'])
    orders = {}; scores = {}; pres = {}; effects = {ARMS[0]: b['effects'].copy()}; choices = {}
    for k, arm in enumerate(ARMS):
        if k == 1:
            new, enough = _forecast(view, b['good'], n)
            for a in ARMS[1:]:
                effects[a] = new[a]
                for i, day in enumerate(meta['dates'][:n]):
                    if day == INITIAL_PARTIAL: effects[a][i] = b['effects'][i]
        chosen, post = study.ep.select_boundary(b['pre'].copy(), b['good'].copy(),
            deepcopy(b['meta']['symbols']), effects[arm].copy(),
            {key: val.copy() for key,val in source.items()}, POLICY.copy())
        post = old._matrix(post, (n, WIDTH), n, 'Final score')
        choices[arm] = [[int(j) for j in row] for row in chosen]
        pres[arm] = b['pre'].copy(); scores[arm] = post
        orders[arm] = study.rp.make_orders(choices[arm], post, b['meta'])
        if k == 0:
            if expected_incumbent_orders is not None:
                require(orders[arm] == expected_incumbent_orders, 'Frozen incumbent ordered choices mismatch')
            if incumbent_validator is not None:
                require(callable(incumbent_validator), 'Incumbent validator callable')
                receipt = incumbent_validator(deepcopy(orders[arm]), post.copy(), b['pre'].copy())
                require(type(receipt) is dict and receipt.get('passed') is True, 'Independent incumbent gate failed')
    events = []; formal = np.asarray([d != INITIAL_PARTIAL for d in meta['dates'][:n]], dtype=bool)
    for i, day in enumerate(meta['dates'][:n]):
        ids = np.flatnonzero(b['good'][i]).tolist()
        rank = sorted(ids, key=lambda j: (-b['pre'][i,j], meta['symbols'][i][j]))
        boundary = [] if len(rank) < 3 else [j for j in rank[2:5]
            if b['pre'][i,rank[2]] - b['pre'][i,j] <= .02]
        for arm in ARMS:
            require(len(choices[arm][i]) == (3 if len(ids)>=3 else 2 if len(ids)==2 else 0) and
                    set(choices[arm][i]) <= set(ids), 'Three/two/cash and original eligibility')
            if not formal[i]:
                require(orders[arm][i] == orders[ARMS[0]][i] and
                    scores[arm][i].tobytes() == scores[ARMS[0]][i].tobytes(), 'Partial exact incumbent')
        events.append(dict(query_index=i, date=day, initial_partial=not bool(formal[i]),
            original_eligible_indices=ids, pre_order=rank, boundary_candidates=boundary,
            episode_budget=float(b['episode'][i]), incumbent_choices=choices[ARMS[0]][i],
            candidate_choices=choices[ARMS[1]][i], control_choices=choices[ARMS[2]][i]))
    counts = dict(events=n, symbols_per_event=WIDTH, initial_partial_events=int((~formal).sum()),
        formal_events=int(formal.sum()), original_eligible_cells=int(b['good'].sum()),
        query_signal_cells=int(((view['query_signal']>0)&formal[:,None]).sum()),
        query_finite_cells=int((_finite_queries(view)&formal[:,None]).sum()),
        calendar_nonzero_effect_cells=int(np.count_nonzero(effects[ARMS[1]][formal])),
        fixed21_nonzero_effect_cells=int(np.count_nonzero(effects[ARMS[2]][formal])),
        calendar_evidence_cells=int(enough[ARMS[1]][formal].sum()),
        fixed21_evidence_cells=int(enough[ARMS[2]][formal].sum()),
        candidate_changed_holdings_events=sum(set(choices[ARMS[0]][i]) != set(choices[ARMS[1]][i]) for i in range(n)),
        control_changed_holdings_events=sum(set(choices[ARMS[0]][i]) != set(choices[ARMS[2]][i]) for i in range(n)),
        candidate_vs_control_changed_holdings_events=sum(set(choices[ARMS[1]][i]) != set(choices[ARMS[2]][i]) for i in range(n)),
        original_episode_calls=1, candidate_episode_calls=1, control_episode_calls=1,
        anchors=len(view['event_dates']), mature_common_anchors=int((view['event_common_valid'] &
            (view['event_common_maturity'] <= view['query_dates'][-1])).sum()) if n else 0,
        new_model_fits=0, predictive_model_fits=0)
    require(tuple(counts)==COUNT_FIELDS and all(tuple(e)==AUDIT_EVENT_KEYS for e in events), 'Decision audit schema')
    require(before == old._snapshot(arrays) and
            metadata == dict({k: meta[k][:n] for k in ('dates','ends','symbols')}, etfs=meta.get('etfs')),
            'Original cache/metadata mutation')
    return orders, scores, pres, effects, dict(schema=SCHEMA, counts=counts, events=events)
