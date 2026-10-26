"""Frozen-core peer comparisons for the existing monthly volatility score.

Pure, no-I/O helpers. Sources and original arrays are supplied by the caller;
neither forecasts, eligibility, core weights nor episode support are modified.
"""
import math

import numpy as np
import pandas as pd

BANDWIDTH = .20
FRACTION = .5


def _sum(values):
    try:
        ans = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Nonfinite comparison sum') from exc
    if not math.isfinite(ans):
        raise FloatingPointError('Nonfinite comparison sum')
    return ans


def _frame(frame):
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError('Adjusted prices require a DataFrame with DatetimeIndex')
    idx = frame.index
    if (idx.hasnans or not idx.is_unique or not idx.is_monotonic_increasing or idx.tz is not None
            or not idx.equals(idx.normalize())):
        raise ValueError('Price sessions must be unique increasing naive dates')
    if not frame.columns.is_unique or any(not isinstance(s, str) or not s or s.strip() != s for s in frame):
        raise ValueError('Price columns must be unique security IDs')


def _day(value):
    day = pd.Timestamp(value)
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError('Expected a naive midnight date')
    return day


def _axes(meta, calendar, limit):
    if not isinstance(meta, dict) or any(k not in meta for k in ('dates', 'symbols', 'etfs')):
        raise ValueError('Metadata requires dates, symbols and etfs')
    dates = [_day(x) for x in calendar]
    if (not dates or any(not isinstance(x, str) or x != str(d.date()) or d.weekday() >= 5
                         for x, d in zip(calendar, dates))
            or any(a >= b for a, b in zip(dates, dates[1:]))):
        raise ValueError('Calendar must contain increasing unique ISO weekday sessions')
    sessions = pd.DatetimeIndex(dates); lookup = {d: i for i, d in enumerate(dates)}
    if len(meta['dates']) != len(meta['symbols']):
        raise ValueError('Event dates and symbols must align')
    if isinstance(meta['etfs'], (str, bytes)) or len(set(meta['etfs'])) != len(meta['etfs']):
        raise ValueError('ETF identities require a unique sequence')
    if any(not isinstance(s, str) or not s or s.strip() != s for s in meta['etfs']):
        raise ValueError('Invalid ETF identity')
    rows, width = [], None
    for date, ss in zip(meta['dates'], meta['symbols']):
        day = _day(date)
        if day not in lookup or (rows and rows[-1][0] >= day):
            raise ValueError('Events must be increasing official sessions')
        if isinstance(ss, (str, bytes)):
            raise ValueError('Symbols require a sequence')
        ss = list(ss)
        if (not ss or len(ss) != len(set(ss)) or
                any(not isinstance(s, str) or not s or s.strip() != s for s in ss)):
            raise ValueError('Event security IDs must be unique')
        if width is not None and len(ss) != width:
            raise ValueError('Event width must be constant')
        width = len(ss); rows.append((day, ss, lookup[day]))
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(rows)):
        raise ValueError('limit must be an integer event prefix')
    return sessions, rows if limit is None else rows[:limit], (len(rows), width or 0)


def _matrix(value, shape, n, name, boolean=False):
    arr = np.asarray(value)
    if arr.shape != shape or (boolean and arr.dtype.kind != 'b'):
        raise ValueError(name + ' must match full event matrix' + (' and be boolean' if boolean else ''))
    try:
        return arr[:n].astype(bool if boolean else float, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be numeric') from exc


def _state(value, rows, n, name):
    arr = np.asarray(value)
    if arr.shape != (rows,) or arr.dtype.kind != 'b':
        raise ValueError(name + ' must be a full boolean event vector')
    return arr[:n]


def _weight(value, shape, n, name):
    arr = np.asarray(value)
    if arr.shape == (shape[0],):
        arr = arr[:, None]
    if arr.shape not in ((shape[0], 1), shape):
        raise ValueError(name + ' shape must be n, n-by-1 or n-by-width')
    ans = np.broadcast_to(arr[:n].astype(float, copy=False), (n, shape[1]))
    if not np.isfinite(ans).all():
        raise ValueError(name + ' must be finite')
    return ans


def _sources(original, source, shape, n):
    required = {'vol', 'vol_weight', 'state_highvol'}
    if not isinstance(source, dict) or not required <= set(source) or set(source)-required-{'core', 'core_weight'}:
        raise ValueError('Source requires vol, vol_weight, state_highvol and optional core/core_weight')
    calm = _state(original['state_calm10'], shape[0], n, 'state_calm10')
    high = _state(source['state_highvol'], shape[0], n, 'state_highvol')
    cw = _weight(original['core_weight'], shape, n, 'core_weight')
    if not np.array_equal(cw, np.broadcast_to(np.where(calm, .5, .52)[:, None], cw.shape)):
        raise ValueError('Original core weight disagrees with calm state')
    core = _matrix(original['core'], shape, n, 'core')
    vol = _matrix(source['vol'], shape, n, 'vol')
    vw = _weight(source['vol_weight'], shape, n, 'vol_weight')
    expected = np.broadcast_to(np.where(calm, .20, .219375*(1.-np.where(high, .10, 0.)/.48))[:, None], vw.shape)
    if not np.array_equal(vw, expected):
        raise ValueError('Original volatility coefficient disagrees with source states')
    if not (np.isfinite(core).all() and np.isfinite(vol).all()):
        raise ValueError('Original core and vol ranks must be finite, including neutral missing ranks')
    if 'core' in source and not np.array_equal(_matrix(source['core'], shape, n, 'source core'), core):
        raise ValueError('Source and original core differ')
    if 'core_weight' in source and not np.array_equal(_weight(source['core_weight'], shape, n, 'source cw'), cw):
        raise ValueError('Source and original core weight differ')
    return core, cw, vol, vw, calm, high


def _rank(values):
    return pd.Series(values).replace([np.inf, -np.inf], np.nan).rank(method='average', pct=True).fillna(.5).to_numpy()-.5


def peer_comparisons(core_rank, raw_vol, original_vol_rank, core_valid):
    """One whole-domain symmetric operator; no eligibility or asset-type input."""
    u, sig, v = (np.asarray(x, float) for x in (core_rank, raw_vol, original_vol_rank))
    valid_core = np.asarray(core_valid)
    if u.ndim != 1 or sig.shape != u.shape or v.shape != u.shape or valid_core.shape != u.shape or valid_core.dtype.kind != 'b':
        raise ValueError('Peer inputs must be aligned vectors and a boolean core mask')
    if not (np.isfinite(u).all() and np.isfinite(v).all()):
        raise ValueError('Original rank vectors must be finite')
    if not np.array_equal(_rank(sig), v):
        raise ValueError('Original volatility rank domain or ties changed')
    domain = np.isfinite(sig); ids = np.flatnonzero(domain); count = len(ids)
    candidate, control = np.full(u.shape, np.nan), np.full(u.shape, np.nan)
    out = dict(domain_count=count, row_ready=False, reason='insufficient_vol_domain',
               mean_peer_weight=None, unordered_pair_weight_sum=0.,
               neighbor_weight_sums=np.zeros(u.shape).tolist(), signed_comparison_sums=np.zeros(u.shape).tolist(),
               candidate_rank_sum=None, control_rank_sum=None, original_rank_sum=None,
               unordered_pair_comparisons=0, nonzero_unordered_pair_comparisons=0, identity_branch=None)
    if count < 2:
        return candidate, control, domain & False, out
    if not valid_core[domain].all():
        out['reason'] = 'missing_core_in_vol_domain'
        return candidate, control, domain & False, out
    weights, signed = [[] for _ in u], [[] for _ in u]
    pair_weights, nonzero = [], 0
    for pos, i in enumerate(ids):
        for j in ids[pos+1:]:
            w = max(1.-abs(float(u[i])-float(u[j]))/BANDWIDTH, 0.)
            sign = float(sig[i] > sig[j])-float(sig[i] < sig[j])
            weights[i].append(w); weights[j].append(w)
            signed[i].append(w*sign); signed[j].append(-w*sign)
            pair_weights.append(w); nonzero += int(w > 0.)
    total = _sum(pair_weights); average = 2.*total/(count*(count-1))
    if not 0. <= average <= 1.:
        raise FloatingPointError('Mean comparison weight outside [0,1]')
    signed_sums = np.array([_sum(x) for x in signed])
    neighbor_sums = np.array([_sum(x) for x in weights])
    offset = .5/count
    candidate[domain] = offset+signed_sums[domain]/(2.*count)
    control[domain] = offset+average*(v[domain]-offset)
    identity = ('all_core_ties' if np.all(u[domain] == u[ids[0]]) else
                'all_vol_ties' if np.all(sig[domain] == sig[ids[0]]) else None)
    if identity is not None:
        candidate[domain] = v[domain]; control[domain] = v[domain]
    if not (np.isfinite(candidate[domain]).all() and np.isfinite(control[domain]).all()):
        raise FloatingPointError('Nonfinite peer rank')
    sums = [_sum(x[domain]) for x in (v, candidate, control)]
    if any(abs(x-.5) > 1e-12 for x in sums):
        raise FloatingPointError('Full-domain rank sum changed')
    out.update(row_ready=True, reason='available', mean_peer_weight=average,
               unordered_pair_weight_sum=total, neighbor_weight_sums=neighbor_sums.tolist(),
               signed_comparison_sums=signed_sums.tolist(), original_rank_sum=sums[0],
               candidate_rank_sum=sums[1], control_rank_sum=sums[2],
               unordered_pair_comparisons=len(pair_weights), nonzero_unordered_pair_comparisons=nonzero,
               identity_branch=identity)
    return candidate, control, domain, out


def _json_numbers(values):
    return [float(x) if np.isfinite(x) else None for x in values]


def build_features(panel, meta, calendar, original, source, limit=None):
    if not isinstance(panel, dict) or set(panel) != {'adj'}:
        raise ValueError('Panel requires exactly adj')
    _frame(panel['adj'])
    sessions, queries, full_shape = _axes(meta, calendar, limit)
    n, width = len(queries), full_shape[1]
    core, cw, v, vw, calm, high = _sources(original, source, full_shape, n)
    values = dict(candidate=np.full((n, width), np.nan), control=np.full((n, width), np.nan),
                  available=np.zeros((n, width), bool))
    history, reasons = [], {}
    if queries:
        last = queries[-1][0]
        prefix = panel['adj'].loc[:last]
        if prefix.empty or 'SPY' not in prefix:
            raise ValueError('Original source requires SPY history')
        needed = sorted({'SPY'} | {s for _, ss, _ in queries for s in ss})
        start = prefix.index[0]
        official = sessions[(sessions >= start) & (sessions <= last)]
        overlap = prefix.index[prefix.index >= sessions[0]]
        if len(overlap.difference(sessions)):
            raise ValueError('Source includes nonofficial covered sessions')
        prefix = prefix.reindex(index=prefix.index.union(official), columns=needed)
        try:
            px = prefix.astype(float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('Price prefix must be numeric') from exc
        x = px.to_numpy(); supplied = ~np.isnan(x)
        if not (np.isfinite(x[supplied]) & (x[supplied] > 0.)).all():
            raise ValueError('Observed prices must be finite positive; NaN is missing')
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            r = px.pct_change(fill_method=None)
        if np.isinf(r.to_numpy()).any():
            raise FloatingPointError('Simple-return arithmetic overflow')
        sigma126 = r.rolling(126).std(ddof=1)
        sigma63 = r.rolling(63).std(ddof=1)
        risk63 = sigma63.replace(0., np.nan)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            calm_core = px.shift(42)/px.shift(252)-1.
            other_core = (px.shift(42)/px.shift(294)-1.)/risk63
        spy_vol = sigma63['SPY']
        rebuilt_calm = spy_vol*np.sqrt(252.) < .10
        rebuilt_high = spy_vol > spy_vol.rolling(252).median()
        for i, (day, symbols, t) in enumerate(queries):
            if bool(rebuilt_calm.at[day]) != bool(calm[i]) or bool(rebuilt_high.at[day]) != bool(high[i]):
                raise ValueError('Original market state parity failed')
            raw_vol = sigma126.loc[day, symbols].to_numpy()
            raw_core = (calm_core if calm[i] else other_core).loc[day, symbols].to_numpy()
            ranked_core = _rank(raw_core); stored_centered_core = core[i]/cw[i]
            if not np.allclose(ranked_core, stored_centered_core, rtol=0., atol=1e-12):
                raise ValueError('Original core rank parity failed')
            if not np.allclose(ranked_core*cw[i], core[i], rtol=0., atol=1e-12):
                raise ValueError('Original weighted core parity failed')
            # Reconstruct the original percentile before distances. Dividing a
            # stored weighted score can perturb tied ranks by one floating bit.
            u = ranked_core+.5
            result = peer_comparisons(u, raw_vol, v[i], np.isfinite(raw_core))
            values['candidate'][i], values['control'][i], values['available'][i], details = result
            reasons[details['reason']] = reasons.get(details['reason'], 0)+1
            history.append(dict(date=str(day.date()), symbols=symbols, session_position=t,
                                calm=bool(calm[i]), highvol=bool(high[i]), core_horizon=252 if calm[i] else 294,
                                raw_vol=_json_numbers(raw_vol), raw_core=_json_numbers(raw_core),
                                core_rank=u.tolist(), vol_rank=v[i].tolist(),
                                vol_domain=np.isfinite(raw_vol).tolist(), core_valid=np.isfinite(raw_core).tolist(),
                                candidate=_json_numbers(values['candidate'][i]), control=_json_numbers(values['control'][i]),
                                **details))
    available = int(values['available'].sum()); ready = sum(row['row_ready'] for row in history)
    return values, dict(events=n, symbols_per_event=width, checked_cells=n*width,
                        source_vol_finite_cells=sum(row['domain_count'] for row in history),
                        available_cells=available, unavailable_cells=n*width-available,
                        ready_rows=ready, fallback_rows=n-ready,
                        unordered_pair_comparisons=sum(row['unordered_pair_comparisons'] for row in history),
                        nonzero_unordered_pair_comparisons=sum(row['nonzero_unordered_pair_comparisons'] for row in history),
                        predictive_model_fits=0, bandwidth=BANDWIDTH, replacement_fraction=FRACTION,
                        reason_counts=reasons, event_history=history, common_availability=True,
                        ranking_eligibility_filtered=False, portfolio_returns_calculated=False)


def build_deltas(values, original, source, limit=None):
    if not isinstance(values, dict) or set(values) != {'candidate', 'control', 'available'}:
        raise ValueError('Values require candidate, control and available')
    available = np.asarray(values['available'])
    full_shape = np.asarray(original['core']).shape
    if (len(full_shape) != 2 or available.ndim != 2 or available.shape[1] != full_shape[1]
            or available.shape[0] > full_shape[0] or available.dtype.kind != 'b'):
        raise ValueError('Availability must match an original event prefix')
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(available)):
        raise ValueError('limit must be an integer prefix of values')
    n = len(available) if limit is None else limit; available = available[:n]
    core, cw, v, vw, calm, high = _sources(original, source, full_shape, n)
    weak = _state(original['state_weak_narrow'], full_shape[0], n, 'weak_narrow')
    extension = _matrix(original['extension_available'], full_shape, n, 'extension', True)
    forecast = _matrix(original['forecast_available'], full_shape, n, 'forecast', True)
    eligible = _matrix(original['eligible'], full_shape, n, 'eligible', True)
    rank = _matrix(original['rank_log_excess'], full_shape, n, 'rank_log_excess')
    active = (calm & ~weak)[:, None] & extension & forecast & eligible & np.isfinite(rank)
    spent = .025*calm[:, None]+.05*weak[:, None]+.01*active
    effective = vw*(1.-spent/(1.-cw))
    if not (np.isfinite(effective).all() and (effective >= 0.).all()):
        raise FloatingPointError('Invalid effective volatility coefficient')
    deltas = {}
    for key in ('candidate', 'control'):
        val = np.asarray(values[key])
        if val.shape != np.asarray(values['available']).shape:
            raise ValueError('Feature value shape changed')
        val = val[:n].astype(float, copy=False)
        if not np.array_equal(np.isfinite(val), available) or np.isinf(val).any():
            raise ValueError('Both arms must use exactly the common finite mask')
        delta = np.zeros(available.shape)
        delta[available] = FRACTION*effective[available]*(val[available]-v[available])
        if n:
            delta[0] = 0.
        if not np.isfinite(delta).all():
            raise FloatingPointError('Nonfinite score correction')
        deltas[key] = delta
    return deltas, dict(events=n, width=full_shape[1], dose=FRACTION, bandwidth=BANDWIDTH,
                        initial_delta_zero=True, scope_active=active.tolist(), base_vol_weights=vw.tolist(),
                        effective_vol_weights=effective.tolist(), available_cells=int(available.sum()),
                        changed_cells={key: int(np.count_nonzero(value)) for key, value in deltas.items()})
