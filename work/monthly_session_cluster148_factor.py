"""Pure skip42 session-event clustering. No I/O, score changes or portfolio work.

``dates`` is the caller-verified complete exchange-session grid. DataFrames may
omit rows/columns; they are reindexed onto each exact window without filling.
Both price inputs must already use the same adjusted basis. This module cannot
certify corporate actions, quote-gap interpretation, or first-public vintages.
"""
from collections import Counter
import numpy as np
import pandas as pd

WINDOW, SKIP, PAIRS, WIDTH = 210, 42, 209, 49
ZERO_TOLERANCE = 1e-12
FIELDS = ('frequency', 'cluster', 'frequency_rank', 'cluster_rank', 'available')
METHOD = 'adjusted_session_event_lag1_covariance_skip42'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _sessions(values, name):
    index = pd.DatetimeIndex(values)
    require(index.tz is None and not index.hasnans and index.is_unique
            and index.is_monotonic_increasing and index.equals(index.normalize()),
            name+' must contain sorted unique timezone-naive session dates')
    return index


def _prices(values, size):
    values = np.asarray(values)
    require(values.shape == (size,), 'Exact price-window length required')
    if values.dtype.kind not in 'fiu':
        require(values.dtype.kind == 'O' and all(
            value is None or value is pd.NA or (
                isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, (bool, np.bool_))) for value in values),
            'Prices must be numeric, not bool/string')
        values = [np.nan if value is None or value is pd.NA else value for value in values]
    return np.asarray(values, dtype=np.float64)


def _empty(reason):
    return dict(available=False, reason=reason, event_count=WINDOW, pair_count=PAIRS,
                event_ones=None, left_ones=None, right_ones=None,
                adjacent_one_pairs=None, covariance_numerator=None,
                frequency=None, cluster=None)


def event_statistics(close, opening):
    """211 closes, 210 opens; final open/close belong to the same final day."""
    close, opening = _prices(close, WINDOW+1), _prices(opening, WINDOW)
    if not ((np.isfinite(close) & (close > 0)).all()
            and (np.isfinite(opening) & (opening > 0)).all()):
        return _empty('invalid_or_missing_window_price')
    with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
        overnight = np.log(opening/close[:-1])
        intraday = np.log(close[1:]/opening)
    if not (np.isfinite(overnight).all() and np.isfinite(intraday).all()):
        return _empty('unrepresentable_log_ratio')
    events = (overnight > ZERO_TOLERANCE) & (intraday < -ZERO_TOLERANCE)
    total = int(np.count_nonzero(events))
    left = int(np.count_nonzero(events[:-1]))
    right = int(np.count_nonzero(events[1:]))
    both = int(np.count_nonzero(events[:-1] & events[1:]))
    numerator = both*PAIRS-left*right
    return dict(available=True, reason='available', event_count=WINDOW, pair_count=PAIRS,
                event_ones=total, left_ones=left, right_ones=right,
                adjacent_one_pairs=both, covariance_numerator=numerator,
                frequency=total/WINDOW, cluster=numerator/(PAIRS*PAIRS))


def centered_lower_rank(values, available):
    """Average tied zero-based rank, lower values preferred; missing ranks zero."""
    values, available = np.asarray(values), np.asarray(available)
    require(values.ndim == available.ndim == 1 and values.shape == available.shape
            and available.dtype == bool, 'Aligned vector and boolean rank mask required')
    require(np.isfinite(values[available]).all(), 'Available rank values must be finite')
    result = np.zeros(values.shape, dtype=np.float64)
    indices = np.flatnonzero(available)
    if len(indices) <= 1:
        return result
    ordered = indices[np.argsort(values[indices], kind='stable')]
    start = 0
    while start < len(ordered):
        stop = start+1
        while stop < len(ordered) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        rank0 = (start+stop-1)/2
        result[ordered[start:stop]] = .5-rank0/(len(ordered)-1)
        start = stop
    return result


def compute_features(dates, close, opening, signal_dates, pools):
    """Return F/C, two centered ranks and a common bool mask, all (signals,49).

    At signal position t: closes[t-252:t-41], opens[t-251:t-41].
    Thus all210 events are t-251..t-42, with209 adjacent event pairs. Invalid
    windows have raw NaNs and zero ranks. Raw price availability alone determines
    this rank universe; original trading eligibility remains caller-owned.
    """
    dates, signals = _sessions(dates, 'dates'), _sessions(signal_dates, 'signals')
    positions = dates.get_indexer(signals)
    require((positions >= 0).all(), 'Signals must be on the supplied session grid')
    require(len(pools) == len(signals), 'One original pool per signal required')
    for frame in (close, opening):
        require(isinstance(frame, pd.DataFrame), 'Named adjusted-price DataFrames required')
        index = _sessions(frame.index, 'price index')
        require(index.isin(dates).all(), 'Price rows outside the supplied session grid')
        require(frame.columns.is_unique and all(type(s) is str and bool(s) for s in frame.columns),
                'Unique named security columns required')
    shape = (len(signals), WIDTH)
    result = {key: np.full(shape, np.nan, dtype=np.float64) for key in ('frequency', 'cluster')}
    result.update({key: np.zeros(shape, dtype=np.float64)
                   for key in ('frequency_rank', 'cluster_rank')})
    result['available'] = np.zeros(shape, dtype=bool)
    cells, months = [], []
    for i, (signal, names, pos) in enumerate(zip(signals, pools, positions)):
        require(len(names) == WIDTH and len(set(names)) == WIDTH
                and all(type(s) is str and bool(s) for s in names), 'Exactly49 unique candidate names')
        enough = pos >= WINDOW+SKIP
        window = dates[pos-WINDOW-SKIP:pos-SKIP+1] if enough else dates[:0]
        c = close.reindex(index=window, columns=names) if enough else None
        o = opening.reindex(index=window[1:], columns=names) if enough else None
        start = len(cells)
        for j, symbol in enumerate(names):
            stats = event_statistics(c[symbol].to_numpy(), o[symbol].to_numpy()) if enough else _empty('insufficient_history')
            result['available'][i, j] = stats['available']
            if stats['available']:
                for key in ('frequency', 'cluster'):
                    result[key][i, j] = stats[key]
            cells.append(dict(signal_index=i, pool_index=j, symbol=symbol,
                              signal_date=str(signal.date()), **stats))
        for key in ('frequency', 'cluster'):
            result[key+'_rank'][i] = centered_lower_rank(result[key][i], result['available'][i])
        for j, cell in enumerate(cells[start:]):
            cell.update(frequency_rank=float(result['frequency_rank'][i, j]),
                        cluster_rank=float(result['cluster_rank'][i, j]))
        months.append(dict(signal_index=i, signal_date=str(signal.date()),
            close_window_start=str(window[0].date()) if enough else None,
            event_window_start=str(window[1].date()) if enough else None,
            window_end=str(window[-1].date()) if enough else None,
            available_candidates=int(np.count_nonzero(result['available'][i])),
            rank_available_candidates_only=True))
    result['audit'] = dict(
        method=METHOD, window_events=WINDOW, adjacent_pairs=PAIRS,
        close_rows=WINDOW+1, opening_rows=WINDOW, skip_sessions=SKIP, width=WIDTH,
        signal_count=len(signals), sign_zero_tolerance=ZERO_TOLERANCE,
        rank_definition='0.5-average_tied_zero_based_rank/(n_valid-1); n_valid<=1 gives0',
        rank_direction='lower_frequency_or_covariance_is_higher',
        raw_missing='NaN', rank_missing=0, common_mask=True,
        constant_event_sequence_valid=True, covariance_uses_two209_pair_marginals=True,
        covariance_arithmetic='integer_joint_count_times209_minus_left_times_right_over209_squared',
        caller_verified_complete_session_grid_required=True,
        caller_verified_common_adjusted_price_basis_required=True,
        etfs_allowed=True, no_trading_eligibility_changes=True, dose_not_applied=True,
        new_model_fits=0, portfolio_returns_or_nav_computed=False,
        reasons=dict(Counter(x['reason'] for x in cells)), months=months, cells=cells)
    return result
