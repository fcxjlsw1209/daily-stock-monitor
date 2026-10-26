"""Pure preparatory monthly close/open target pairing; no market-data loading.

The caller supplies an already provenance-validated, complete official session
calendar. A DatetimeIndex alone cannot prove that an interior session was not
omitted. This adapter validates axes, required month boundaries and maturity;
it never fills a missing price or substitutes the final visible price session.
Existing common-good retains upstream full-path restrictions. Only endpoint
values of its eligible, feature-finite rows are inspected here.
"""
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd

_VALIDATOR_PATH = Path(__file__).resolve().parents[1] / 'outputs/monthly_target_definition70/labels.py'
_SPEC = importlib.util.spec_from_file_location('after142_old_target_validation', _VALIDATOR_PATH)
old = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(old)

ARMS = ('paired_close', 'paired_open')
MIN_MONTH_ROWS = 2
CLIP_LIMIT = 1.0
LOOKBACK_YEARS = 5


def _calendar_sessions(official_calendar):
    value = official_calendar if isinstance(official_calendar, pd.DatetimeIndex) else getattr(official_calendar, 'sessions', None)
    if not isinstance(value, pd.DatetimeIndex):
        raise ValueError('official_calendar must be a DatetimeIndex or expose DatetimeIndex sessions')
    days = old._days(value, 'official sessions')
    sessions = pd.DatetimeIndex(days)
    if not len(sessions) or not sessions.is_unique or not sessions.is_monotonic_increasing:
        raise ValueError('official sessions must be nonempty, unique and increasing')
    return sessions


def _schedule(official_calendar, dates):
    sessions = _calendar_sessions(official_calendar)
    dates = old._days(dates, 'signal dates')
    n = len(dates)
    calendar_ends = np.empty(n, dtype='datetime64[D]')
    entries = np.empty(n, dtype='datetime64[D]')
    exits = np.empty(n, dtype='datetime64[D]')
    calendar_months = sessions.to_period('M')
    for date in np.unique(dates):
        signal = pd.Timestamp(date)
        month = signal.to_period('M')
        current = sessions[calendar_months == month]
        following = sessions[calendar_months == month + 1]
        if signal not in sessions or not len(current) or signal != current[-1]:
            raise ValueError('each training signal must be an official month-end session')
        if not len(following):
            raise ValueError('official calendar does not cover the full next target month')
        end = following[-1]
        entry_pos = int(sessions.searchsorted(signal, side='right'))
        exit_pos = int(sessions.searchsorted(end, side='right'))
        if entry_pos >= len(sessions) or exit_pos >= len(sessions):
            raise ValueError('official calendar must extend through the session after target month-end')
        entry, exit_session = sessions[entry_pos], sessions[exit_pos]
        if entry.to_period('M') != month + 1 or exit_session.to_period('M') != month + 2:
            raise ValueError('official calendar has a missing required monthly boundary')
        selected = dates == date
        calendar_ends[selected] = end.to_datetime64().astype('datetime64[D]')
        entries[selected] = entry.to_datetime64().astype('datetime64[D]')
        exits[selected] = exit_session.to_datetime64().astype('datetime64[D]')
    return dates, calendar_ends, entries, exits


def validate_maturity_interface(official_calendar, dates, calendar_ends, entry_sessions,
                                exit_sessions, available_sessions):
    """Validate explicit calendar anchor and shared information-available date.

    Both labels are intentionally withheld until the exit opening is known.
    ``calendar_ends`` remains in signal month+1, while exit/availability is in
    month+2. All outputs are owned copies. No price arrays are accessed.
    """
    dates, expected_end, expected_entry, expected_exit = _schedule(official_calendar, dates)
    supplied = {}
    for name, value, expected in (
            ('calendar_ends', calendar_ends, expected_end),
            ('entry_sessions', entry_sessions, expected_entry),
            ('exit_sessions', exit_sessions, expected_exit),
            ('available_sessions', available_sessions, expected_exit)):
        value = old._days(value, name)
        if value.shape != dates.shape or not np.array_equal(value, expected):
            raise ValueError(name + ' differs from the explicit official calendar/maturity contract')
        supplied[name] = value.copy()
    supplied['dates'] = dates.copy()
    return supplied


def mature_training_mask(official_calendar, dates, calendar_ends, entry_sessions,
                         exit_sessions, available_sessions, paired_good, cutoff):
    """Past-five-year shared mask; annual callers supply prior December31.

    The helper also accepts intermediate cutoffs for pure maturity tests. It
    does not fit a model, enforce a minimum sample size, or inspect future data.
    """
    schedule = validate_maturity_interface(official_calendar, dates, calendar_ends,
                                          entry_sessions, exit_sessions, available_sessions)
    good = old._bool(paired_good, len(schedule['dates']), 'paired_good')
    limit = old._days([cutoff], 'cutoff')[0]
    start = (pd.Timestamp(limit) - pd.DateOffset(years=LOOKBACK_YEARS)).to_datetime64().astype('datetime64[D]')
    return (good & (schedule['dates'] > start) & (schedule['available_sessions'] <= limit)).copy()


def _validate_frames(close, opening, dates, ends, symbols, eligible, common_good, finite_features):
    result = old._validate(close, dates, ends, symbols, eligible, common_good, finite_features)
    old._validate(opening, dates, ends, symbols, eligible, common_good, finite_features)
    if not close.index.equals(opening.index) or not close.columns.equals(opening.columns):
        raise ValueError('adjusted close/opening axes must be exactly aligned')
    return result


def _log_excess(asset, benchmark):
    # Log differences avoid overflow/underflow of otherwise positive finite
    # price ratios. No nonfinite result is converted into a zero label.
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        value = (np.log(asset[1]) - np.log(asset[0])) - (np.log(benchmark[1]) - np.log(benchmark[0]))
    if not np.isfinite(value):
        raise FloatingPointError('nonfinite paired log excess')
    return float(value)


def build_paired_targets(adjusted_close, adjusted_open, official_calendar, dates, symbols,
                         original_common_good, original_eligible, original_features_finite):
    """Build only two matched monthly labels from explicitly supplied inputs.

    Missing, nonpositive or nonfinite endpoint observations remove that row
    from BOTH targets, subject to the inherited common-good mask. Missing
    required calendar coverage is an error, not an opportunity to shorten a
    holding interval. Clip to [-1,1], then center among >=2 paired rows/month;
    a centered label can exceed1 in magnitude. No query eligibility is present
    in this interface, and none of the supplied inputs is mutated.
    """
    dates, ends, entries, exits = _schedule(official_calendar, dates)
    dates, ends, symbols, months, masks = _validate_frames(
        adjusted_close, adjusted_open, dates, ends, symbols,
        original_eligible, original_common_good, original_features_finite)
    eligible, common_good, finite_features = masks
    base_good = eligible & common_good & finite_features
    n = len(dates)
    raw = {arm: np.full(n, np.nan) for arm in ARMS}
    targets = {arm: np.full(n, np.nan) for arm in ARMS}
    endpoint_good = np.zeros(n, dtype=bool)
    good = np.zeros(n, dtype=bool)
    audit = []
    for month in np.unique(months):
        rows = np.flatnonzero(months == month)
        candidates = rows[base_good[rows]]
        first = rows[0]
        points = [dates[first], ends[first], entries[first], exits[first]]
        positions = adjusted_close.index.get_indexer(pd.DatetimeIndex(points))
        reason = 'fewer_than_two_paired_rows'
        if not len(candidates):
            reason = 'no_originally_usable_rows'
        elif (positions < 0).any():
            reason = 'missing_price_session'
        else:
            market_close = old._prices(adjusted_close, positions[:2], 'SPY')
            market_open = old._prices(adjusted_open, positions[2:], 'SPY')
            if market_close is None or market_open is None:
                reason = 'invalid_benchmark_endpoint'
            else:
                for row in candidates:
                    stock_close = old._prices(adjusted_close, positions[:2], symbols[row])
                    stock_open = old._prices(adjusted_open, positions[2:], symbols[row])
                    if stock_close is None or stock_open is None:
                        continue
                    raw['paired_close'][row] = _log_excess(stock_close, market_close)
                    raw['paired_open'][row] = _log_excess(stock_open, market_open)
                    endpoint_good[row] = True
        selected = rows[endpoint_good[rows]]
        clips = {arm: 0 for arm in ARMS}
        if len(selected) >= MIN_MONTH_ROWS:
            for arm in ARMS:
                values = raw[arm][selected]
                clips[arm] = int(np.count_nonzero(np.abs(values) > CLIP_LIMIT))
                clipped = np.clip(values, -CLIP_LIMIT, CLIP_LIMIT)
                targets[arm][selected] = clipped - clipped.mean()
            good[selected] = True
            reason = 'complete'
        audit.append(dict(month=str(month), signal_date=str(dates[first]),
                          calendar_end=str(ends[first]), entry_session=str(entries[first]),
                          exit_session=str(exits[first]), available_session=str(exits[first]),
                          input_rows=int(len(rows)), base_rows=int(len(candidates)),
                          endpoint_rows=int(len(selected)), paired_rows=int(good[rows].sum()),
                          reason=reason, clip_counts=clips))
    schedule = validate_maturity_interface(official_calendar, dates, ends, entries, exits, exits)
    return dict(targets=targets, raw_targets=raw, good=good, base_good=base_good.copy(),
                endpoint_good=endpoint_good, symbols=symbols.copy(), **schedule,
                audit=dict(stage='preparatory_paired_labels_only', arms=list(ARMS),
                           min_month_rows=MIN_MONTH_ROWS, clip_limit=CLIP_LIMIT,
                           clipping='clip raw values then center on identical paired monthly rows',
                           maturity='both arms available only at exit_session opening',
                           inherited_full_path_mask=True, future_query_eligibility_used=False,
                           calendar_provenance_verified_here=False, months=audit))
