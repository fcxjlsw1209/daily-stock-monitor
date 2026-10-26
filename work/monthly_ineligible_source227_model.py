"""Source-only expansion within the EXISTING monthly query IDs and cached X.

No strategy, ranks, orders, returns, raw-price acquisition or file I/O.  The
original training cohort/targets never consume the added labels.  Forecasts
reuse frozen Local70 algorithms; the two replay domains have explicit costs.
"""
from pathlib import Path
import importlib.util
from collections import Counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

local = _load('ineligible227_local', ROOT/'outputs/monthly_local_linear70/model.py')
old_labels = _load('ineligible227_labels', ROOT/'outputs/monthly_target_definition70/labels.py')
TARGETS = tuple(old_labels.ARMS)
RAW_KEYS = ('source_log', 'local_constant', 'local_linear')
SOURCE_KEYS = {'raw_source_log', 'raw_local_constant', 'raw_local_linear',
               'raw_candidate', 'raw_control', 'available', 'local_available'}
COUNT_FIELDS = ('query_rows', 'original_available_rows', 'expansion_requested_rows',
    'added_available_rows', 'available_rows', 'old_source_target_fits',
    'new_source_target_fits', 'old_pooled_log_refits', 'new_pooled_log_refits',
    'old_local_evaluations', 'new_local_evaluations', 'old_local_solves',
    'new_local_solves', 'actual_regression_solves')
LABEL_COUNT_FIELDS = ('training_rows', 'original_good_rows', 'added_rows',
    'extended_good_rows', 'original_center_months', 'original_strict_exceptions',
    'new_candidate_rows')
LABEL_PARITY_ATOL = 1e-12


def dependency_files():
    return tuple(sorted(set(local.dependency_files()) | {Path(old_labels.__file__).resolve(),
                                                        Path(__file__).resolve()}))


def _bool(value, n, name):
    value = np.asarray(value)
    if value.shape != (n,) or value.dtype.kind != 'b':
        raise ValueError(name+' must be an exact boolean vector')
    return value


def _exact(left, right, name):
    left, right = np.asarray(left), np.asarray(right)
    if left.dtype != right.dtype or left.shape != right.shape or left.tobytes() != right.tobytes():
        raise AssertionError('Exact original replay failed: '+name)


def _replay_counts(audit):
    source = sum(r['status'] == 'fit' and r['prediction_rows'] > 0
                 for r in audit['source']['annual'])
    pooled = sum(r['prediction_rows'] > 0 for r in audit['annual'])
    return (4*source, pooled, len(audit['local']),
            sum(r['status'] == 'fit' for r in audit['local']))


def forecasts(dates, ends, symbols, X, targets, common_good, event_dates,
              event_symbols, event_X, event_good, *, cached_source, expansion_allowed):
    """Replay old rows exactly, then evaluate only the separately admitted rows.

    expansion_allowed is a caller-verified 295-official-session positive-price
    mask. It never removes old rows. Query labels are not accepted by this API.
    """
    q = len(event_dates)
    old_good = _bool(event_good, q, 'event_good')
    allowed = _bool(expansion_allowed, q, 'expansion_allowed')
    xx = np.asarray(event_X, dtype=float)
    if xx.shape != (q, 10):
        raise ValueError('Exactly ten cached features per query are required')
    if set(cached_source) != SOURCE_KEYS:
        raise ValueError('Exact seven-field frozen206 source required')
    available = _bool(cached_source['available'], q, 'cached available')
    local_available = _bool(cached_source['local_available'], q, 'cached local_available')
    if np.any(available & ~old_good) or np.any(local_available & ~available):
        raise ValueError('Cached availability exceeds original query domain')
    for key in SOURCE_KEYS:
        if np.asarray(cached_source[key]).shape != (q,):
            raise ValueError('Cached source row axis differs: '+key)
    args = (dates, ends, symbols, X, targets, common_good, event_dates, event_symbols, event_X)
    old = local.annual_forecasts(*args, old_good)
    for key in RAW_KEYS:
        _exact(old['raw'][key], cached_source['raw_'+key], key)
    for key in ('available', 'local_available'):
        _exact(old[key], cached_source[key], key)
    new_mask = allowed & np.isfinite(xx).all(axis=1) & ~old_good
    new = local.annual_forecasts(*args, new_mask)
    if np.any(new['available'] & available):
        raise AssertionError('Old and added forecast domains overlap')
    raw = {key: np.asarray(cached_source['raw_'+key]).copy() for key in RAW_KEYS}
    for key in RAW_KEYS:
        raw[key][new['available']] = new['raw'][key][new['available']]
    final_available = available | new['available']
    final_local = local_available | new['local_available']
    for key in RAW_KEYS:
        if not np.array_equal(np.isfinite(raw[key]), final_available):
            raise FloatingPointError('Extended finite availability differs: '+key)
    oc, nc = _replay_counts(old['audit']), _replay_counts(new['audit'])
    counts = dict(query_rows=q, original_available_rows=int(available.sum()),
        expansion_requested_rows=int(new_mask.sum()), added_available_rows=int(new['available'].sum()),
        available_rows=int(final_available.sum()), old_source_target_fits=oc[0],
        new_source_target_fits=nc[0], old_pooled_log_refits=oc[1], new_pooled_log_refits=nc[1],
        old_local_evaluations=oc[2], new_local_evaluations=nc[2], old_local_solves=oc[3],
        new_local_solves=nc[3], actual_regression_solves=oc[0]+nc[0]+oc[1]+nc[1]+oc[3]+nc[3])
    audit = dict(method='same_query_ineligible_prediction_extension', query_rows=q, feature_count=10,
        raw_keys=list(RAW_KEYS), old_parity_exact=True, original_forecasts_copied=True,
        new_queries_separate_replay=True, original_training_unchanged=True,
        features_recomputed=False, reranked=False, expansion_allowed=allowed.tolist(),
        new_query_mask=new_mask.tolist(), old_replay=old['audit'], new_replay=new['audit'], counts=counts)
    return dict(raw=raw, available=final_available, local_available=final_local, audit=audit)


def _calendar(calendar):
    values = list(calendar)
    for s in values:
        if not isinstance(s, str) or len(s) != 10 or pd.Timestamp(s).strftime('%Y-%m-%d') != s:
            raise ValueError('Calendar must use canonical ISO dates')
    index = pd.DatetimeIndex(values)
    if index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError('Official calendar must be unique and increasing')
    return index


def _metadata(train, meta):
    required = ('dates','ends','symbols')
    if any(k not in meta for k in required) or not len(meta['dates']) == len(meta['ends']) == len(meta['symbols']):
        raise ValueError('Training metadata axes differ')
    widths = np.array([len(s) for s in meta['symbols']], dtype=int)
    if any(len(set(s)) != len(s) or any(not isinstance(x, str) or not x or x != x.strip() for x in s)
           for s in meta['symbols']):
        raise ValueError('Explicit unique training security IDs required')
    offsets = np.r_[0, np.cumsum(widths)]
    if np.asarray(train['offsets']).dtype.kind not in 'iu' or not np.array_equal(train['offsets'], offsets):
        raise ValueError('Training offsets differ')
    dates = old_labels._days(train['dates'], 'dates')
    ends = old_labels._days(train['ends'], 'ends')
    for key, value in (('dates', dates), ('ends', ends)):
        if not np.array_equal(value, np.repeat(np.asarray(meta[key], dtype='datetime64[D]'), widths)):
            raise ValueError('Exact training identity differs: '+key)
    symbols = np.asarray([s for row in meta['symbols'] for s in row], dtype=str)
    n = len(symbols)
    X = np.asarray(train['train_plain'], dtype=float)
    if X.shape != (n, 10):
        raise ValueError('Original plain10 dimensions differ')
    eligible = _bool(train['train_eligible'], n, 'train_eligible')
    cached_good = _bool(train['train_good'], n, 'train_good')
    good = _bool(train['common_good'], n, 'common_good')
    for key in TARGETS:
        if np.asarray(train['target_'+key]).shape != (n,):
            raise ValueError('Cached target axis differs: '+key)
    if np.asarray(train['vol63']).shape != (n,):
        raise ValueError('Cached volatility axis differs')
    return dates, ends, symbols, X, eligible, cached_good, good


def _source_prices(adj, dates, symbol):
    if symbol not in adj.columns:
        return None
    positions = adj.index.get_indexer(dates)
    if np.any(positions < 0):
        return None
    values = adj.iloc[positions][symbol].to_numpy()
    if any(isinstance(x, (bool, np.bool_)) for x in values):
        return None
    try:
        values = values.astype(float)
    except (TypeError, ValueError, OverflowError):
        return None
    return values if np.all(np.isfinite(values) & (values > 0)) else None


def _raw(history, asset, spy):
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        returns = history[1:] / history[:-1] - 1.
        vol = float(np.std(returns, ddof=1))
        gross = asset[-1] / asset[0]
        asset_return = gross - 1.
        spy_gross = spy[-1] / spy[0]
        excess = asset_return - (spy_gross - 1.)
        raw = dict(simple_excess=float(excess), log_excess=float(np.log(gross)-np.log(spy_gross)),
            vol_scaled=float(excess/(vol*old_labels.MONTH_SCALE)) if vol > 0 else None,
            loss_utility=float(excess-old_labels.LOSS_PENALTY*max(0.,-asset_return)))
    if not np.isfinite(returns).all() or not np.isfinite(vol) or any(v is not None and not np.isfinite(v) for v in raw.values()):
        raise FloatingPointError('Nonfinite label arithmetic')
    return vol, raw


def _strict(adj, sessions, start, end, symbol):
    pos = sessions.get_indexer([start, end])
    if np.any(pos < 0) or pos[1] <= pos[0]:
        raise ValueError('Label endpoints must be official sessions')
    si, ei = map(int, pos)
    if (si+1 >= len(sessions) or ei+1 >= len(sessions)
        or sessions[si+1].to_period('M') == start.to_period('M')
        or sessions[ei+1].to_period('M') == end.to_period('M')
        or end.to_period('M') != start.to_period('M')+1):
        raise ValueError('Complete next official calendar month must be proved')
    if si < 294:
        return 'insufficient_history', None, None
    if symbol not in adj.columns:
        return 'missing_symbol', None, None
    past, future = sessions[si-294:si+1], sessions[si:ei+1]
    if np.any(adj.index.get_indexer(past) < 0) or np.any(adj.index.get_indexer(future) < 0):
        return 'missing_official_price_session', None, None
    history = _source_prices(adj, past, symbol)
    if history is None:
        return 'invalid_history_price', None, None
    ap = _source_prices(adj, future, symbol)
    if ap is None:
        return 'invalid_future_asset_price', None, None
    if 'SPY' not in adj.columns:
        return 'missing_spy', None, None
    sp = _source_prices(adj, future, 'SPY')
    if sp is None:
        return 'invalid_future_spy_price', None, None
    vol, raw = _raw(history[-64:], ap, sp)
    if vol == 0.:
        return 'zero_volatility', vol, None
    return None, vol, raw


def _old_raw(adj, date, end, symbol):
    """Frozen original label arithmetic: original frame64 and endpoint pairs."""
    start, finish = adj.index.get_indexer([date, end])
    if start < 63 or finish < 0:
        raise AssertionError('Original center source disappeared')
    history = old_labels._prices(adj, slice(start-63,start+1), symbol)
    ap = old_labels._prices(adj, [start,finish], symbol)
    sp = old_labels._prices(adj, [start,finish], 'SPY')
    if history is None or ap is None or sp is None:
        raise AssertionError('Original center prices invalid')
    vol, raw = _raw(history, ap, sp)
    if vol <= 0:
        raise AssertionError('Original center risk invalid')
    return raw


def _json_number(value):
    return float(value) if np.isfinite(value) else None


def build_labels(adj, train, training_meta, calendar):
    """Extend ONLY previously ineligible label rows; never alter training targets.

    Existing labels are checked at predeclared atol1e-12, rtol0, then copied
    exactly. Added labels use the ORIGINAL common cohort's fixed clipped means.
    """
    dates, ends, symbols, X, eligible, cached_good, original_good = _metadata(train, training_meta)
    finite = np.isfinite(X).all(axis=1)
    sessions = _calendar(calendar)
    old = old_labels.build_targets(adj, dates, ends, symbols, eligible, cached_good, finite)
    if not np.array_equal(old['good'], original_good):
        raise AssertionError('Original common label cohort changed')
    targets = {}
    for key in TARGETS:
        expected = np.asarray(train['target_'+key])
        np.testing.assert_allclose(old['targets'][key], expected, atol=LABEL_PARITY_ATOL, rtol=0., equal_nan=True)
        targets[key] = expected.copy()
    np.testing.assert_allclose(old['vol63'], train['vol63'], atol=LABEL_PARITY_ATOL, rtol=0., equal_nan=True)
    vol63 = np.asarray(train['vol63']).copy()
    good, added = original_good.copy(), np.zeros(len(dates), bool)
    months, records = [], []
    for month in np.unique(dates.astype('datetime64[M]')):
        rows = np.flatnonzero(dates.astype('datetime64[M]') == month)
        date, end = pd.Timestamp(dates[rows[0]]), pd.Timestamp(ends[rows[0]])
        original_rows = rows[original_good[rows]]
        centers = {key: None for key in TARGETS}
        if len(original_rows):
            if len(original_rows) < 2:
                raise AssertionError('Original center must contain at least two rows')
            raw = [_old_raw(adj, date, end, symbols[i]) for i in original_rows]
            for key in TARGETS:
                clipped = np.clip(np.array([r[key] for r in raw]), -old_labels.CLIP_LIMITS[key], old_labels.CLIP_LIMITS[key])
                centers[key] = float(clipped.mean())
        for i in rows:
            strict_valid, raw, exception = None, None, False
            if eligible[i] and not original_good[i]:
                reason = 'original_eligible_not_admitted'
            elif not finite[i]:
                reason = 'nonfinite_features'; strict_valid = False
            else:
                source_reason, v, raw = _strict(adj, sessions, date, end, symbols[i])
                strict_valid = source_reason is None
                if original_good[i]:
                    exception = source_reason is not None
                    reason = 'original_retained_exception:'+source_reason if exception else 'original_retained'
                elif source_reason is not None:
                    reason = source_reason
                else:
                    vol63[i] = v
                    if len(original_rows) < 2:
                        reason = 'fewer_than_two_original_center_rows'
                    else:
                        for key in TARGETS:
                            targets[key][i] = float(np.clip(raw[key], -old_labels.CLIP_LIMITS[key], old_labels.CLIP_LIMITS[key])-centers[key])
                        good[i] = added[i] = True
                        reason = 'added'
            records.append(dict(row=int(i), symbol=str(symbols[i]), date=str(dates[i]), end=str(ends[i]),
                original_eligible=bool(eligible[i]), original_good=bool(original_good[i]), added=bool(added[i]),
                strict_source_valid=strict_valid, reason=reason, original_strict_exception=exception,
                vol63=_json_number(vol63[i]), raw_targets=raw))
        months.append(dict(date=str(date.date()),end=str(end.date()),rows=rows.tolist(),
            original_center_row_ids=original_rows.tolist(),centers=centers,
            original_good_rows=len(original_rows),added_rows=int(added[rows].sum())))
    records.sort(key=lambda r:r['row'])
    counts = dict(training_rows=len(dates), original_good_rows=int(original_good.sum()),
        added_rows=int(added.sum()), extended_good_rows=int(good.sum()),
        original_center_months=sum(m['original_good_rows'] >= 2 for m in months),
        original_strict_exceptions=sum(r['original_strict_exception'] for r in records),
        new_candidate_rows=int((~eligible).sum()))
    audit = dict(method='same_query_ineligible_label_extension',label_parity_atol=LABEL_PARITY_ATOL,
        label_parity_rtol=0.,original_targets_copied=True,original_common_good_preserved=True,
        original_centers_only=True,added_rows_never_train=True,full_official_paths_required_for_added=True,
        original_strict_exceptions_retained=True,counts=counts,
        reason_counts=dict(sorted(Counter(r['reason'] for r in records).items())),months=months,rows=records)
    return dict(targets=targets,good=good,added=added,vol63=vol63,audit=audit)
