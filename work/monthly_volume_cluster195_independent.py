"""Independent scalar volume-event oracle; no production feature imports.

The only reusable implementation is frozen independent portfolio accounting.
Stored volume is constant-unit invariant, not split-adjusted turnover: a
within-window unit jump can change the strict median events.
"""
from datetime import date
from fractions import Fraction
import hashlib
import importlib.util
import math
import numbers
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN_HELPER = ROOT / 'work/monthly_sector_recurrence192_independent.py'
FROZEN_HELPER_SHA = 'f78cc005827fd977ec431bdec4af0f4b7160694cb3d382df0115e8e00e0bedf8'
ARMS = ('incumbent_local67', 'volume_cluster', 'volume_frequency_control')
RAW = {'volume_cluster': 'candidate', 'volume_frequency_control': 'control'}
DOSE = .01


def require(ok, message):
    if not ok:
        raise ValueError(message)


def frozen_helper():
    require(hashlib.sha256(FROZEN_HELPER.read_bytes()).hexdigest() == FROZEN_HELPER_SHA,
            'Frozen192 independent accounting helper changed')
    spec = importlib.util.spec_from_file_location('volume195_frozen_independent_accounts', FROZEN_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependency_files():
    return tuple(sorted(set(frozen_helper().dependency_files()) | {Path(__file__).resolve()}))


def accounting():
    return frozen_helper().accounting()


def audit_outcomes(px, opening, orders, exports, results, meta):
    helper = frozen_helper()
    helper.ARMS = ARMS
    receipt = helper.audit_outcomes(px, opening, orders, exports, results, meta)
    require(receipt['checked_paths'] == 9 and receipt['checked_slices'] == 45 and
            set(receipt['paths']) == set(ARMS), 'Incomplete nine-path accounting')
    receipt['outcome_adapter_sha256'] = FROZEN_HELPER_SHA
    return receipt


def record_check(actual, expected, label='audit'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ': keys')
        for key in expected:
            record_check(actual[key], expected[key], label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list')
        for i, (a, b) in enumerate(zip(actual, expected)):
            record_check(a, b, label + '/' + str(i))
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': value/type')


def iso(value):
    require(isinstance(value, str), 'ISO date string required')
    parsed = date.fromisoformat(value)
    require(parsed.isoformat() == value, 'Strict ISO date required')
    return pd.Timestamp(parsed)


def schema(panel, meta, calendar):
    require(isinstance(panel, dict) and set(panel) == {'volume'}, 'Panel keys differ')
    frame = panel['volume']
    require(isinstance(frame, pd.DataFrame) and isinstance(frame.index, pd.DatetimeIndex), 'Volume frame type')
    idx = frame.index
    require(not idx.hasnans and idx.is_unique and idx.is_monotonic_increasing and idx.tz is None and
            idx.equals(idx.normalize()), 'Invalid volume daily axis')
    require(frame.columns.is_unique and all(isinstance(s, str) and s and s.strip() == s
            for s in frame.columns), 'Invalid volume symbol labels')
    require(isinstance(meta, dict) and all(k in meta for k in ('dates', 'symbols', 'etfs')), 'Metadata fields')
    require(len(meta['dates']) == len(meta['symbols']), 'Event metadata lengths')
    require(isinstance(meta['etfs'], (list, tuple, set, frozenset)) and
            len(meta['etfs']) == len(set(meta['etfs'])) and
            all(isinstance(s, str) and s and s.strip() == s for s in meta['etfs']), 'ETF identities')
    sessions = [iso(x) for x in calendar]
    require(sessions and all(d.weekday() < 5 for d in sessions) and
            all(a < b for a, b in zip(sessions, sessions[1:])), 'Calendar order/sessions')
    positions = {d: i for i, d in enumerate(sessions)}
    queries, width = [], 0
    for value, symbols in zip(meta['dates'], meta['symbols']):
        day = iso(value)
        require(day in positions and (not queries or day > queries[-1][0]), 'Event session/order')
        t = positions[day]
        if queries or value != '2016-08-17':
            require(t + 1 < len(sessions) and sessions[t + 1].to_period('M') != day.to_period('M'),
                    'Verified official month-end required')
        require(isinstance(symbols, (list, tuple)) and symbols and len(symbols) == len(set(symbols)),
                'Event identities')
        require(all(isinstance(s, str) and s and s.strip() == s for s in symbols), 'Invalid event symbol')
        require(not queries or width == len(symbols), 'Changing event width')
        width = len(symbols)
        queries.append((day, list(symbols), t))
    return sessions, queries, width


def empty(reason):
    return dict(available=False, reason=reason, candidate=None, control=None, median_values=[],
        high_volume_events=[], event_ones=None, left_ones=None, right_ones=None, adjacent_one_pairs=None,
        covariance_numerator=None, event_observations=0, adjacent_pair_observations=0, rolling_medians_computed=0)


def measure_oracle(volume):
    require(np.shape(volume) == (273,), 'Exactly273 stored volume observations required')
    raw = list(volume)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real) for v in raw):
        return empty('nonnumeric_volume')
    try:
        numeric = [float(x) for x in raw]
    except (ValueError, TypeError, OverflowError):
        return empty('nonnumeric_volume')
    if any(not math.isfinite(x) or x <= 0 for x in numeric):
        return empty('invalid_volume')
    medians, events = [], []
    for position in range(63, 273):
        previous = sorted(numeric[position - 63:position])
        median = previous[31]
        medians.append(median)
        events.append(numeric[position] > median)
    total = sum(events)
    left, right = sum(events[:-1]), sum(events[1:])
    both = sum(a and b for a, b in zip(events[:-1], events[1:]))
    numerator = 209 * both - left * right
    return dict(available=True, reason='available', candidate=float(Fraction(numerator, 209 * 209)),
        control=float(Fraction(total, 210)), median_values=medians, high_volume_events=events,
        event_ones=total, left_ones=left, right_ones=right, adjacent_one_pairs=both,
        covariance_numerator=numerator, event_observations=210, adjacent_pair_observations=209,
        rolling_medians_computed=210)


def centered_ranks(raw, available):
    out = np.zeros(np.shape(raw), float)
    for i in range(len(out)):
        columns = [j for j, good in enumerate(available[i]) if good]
        n = len(columns)
        if n <= 1:
            continue
        for j in columns:
            lesser = sum(raw[i, k] < raw[i, j] for k in columns)
            equal = sum(raw[i, k] == raw[i, j] for k in columns)
            rank0 = lesser + (equal - 1) / 2
            out[i, j] = rank0 / (n - 1) - .5
    return out


def independent_features(panel, meta, calendar, limit=None):
    sessions, queries, width = schema(panel, meta, calendar)
    require(limit is None or type(limit) is int and 0 <= limit <= len(queries), 'Invalid prefix limit')
    if limit is not None:
        queries = queries[:limit]
    shape = (len(queries), width)
    values = dict(candidate=np.full(shape, np.nan), control=np.full(shape, np.nan),
                  available=np.zeros(shape, bool), candidate_rank=np.zeros(shape), control_rank=np.zeros(shape))
    history, reasons = [], {}
    frame = panel['volume']
    for i, (day, names, t) in enumerate(queries):
        dates = sessions[t - 314:t - 41] if t >= 314 else []
        missing = [str(d.date()) for d in dates if d not in frame.index]
        cells = []
        for j, symbol in enumerate(names):
            reason = ('insufficient_history' if not dates else 'missing_symbol' if symbol not in frame.columns
                      else 'missing_official_session' if missing else None)
            cell = empty(reason) if reason else measure_oracle([frame.at[d, symbol] for d in dates])
            cell.update(symbol=symbol, date=str(day.date()), candidate_rank=0., control_rank=0.)
            values['available'][i, j] = cell['available']
            values['candidate'][i, j] = cell['candidate']
            values['control'][i, j] = cell['control']
            reasons[cell['reason']] = reasons.get(cell['reason'], 0) + 1
            cells.append(cell)
        history.append(dict(date=str(day.date()), symbols=names, session_position=t,
            volume_start=str(dates[0].date()) if dates else None,
            volume_end=str(dates[-1].date()) if dates else None,
            event_start=str(dates[63].date()) if dates else None,
            event_end=str(dates[-1].date()) if dates else None,
            missing_official_sessions=missing, cells=cells))
    for raw in ('candidate', 'control'):
        key = raw + '_rank'
        values[key] = centered_ranks(values[raw], values['available'])
        for i, event in enumerate(history):
            for j, cell in enumerate(event['cells']):
                cell[key] = float(values[key][i, j])
    available = int(values['available'].sum())
    audit = dict(events=len(queries), symbols_per_event=width, checked_cells=int(values['available'].size),
        volume_count=273, skip_sessions=42, event_count=210, pair_count=209, median_window=63,
        available_cells=available, unavailable_cells=int(values['available'].size) - available,
        reason_counts=reasons, event_observations=210 * available, adjacent_pair_observations=209 * available,
        rolling_medians_computed=210 * available, predictive_model_fits=0, new_model_fits=0,
        common_availability=True, ranking_performed=True, ranking_eligibility_filtered=False,
        trading_eligibility_modified=False, portfolio_returns_calculated=False, event_history=history)
    return values, audit


def arrays(values):
    require(isinstance(values, dict) and set(values) ==
            {'candidate', 'control', 'available', 'candidate_rank', 'control_rank'}, 'Feature keys differ')
    a = {key: np.asarray(value) for key, value in values.items()}
    mask = a['available']
    require(mask.ndim == 2 and mask.dtype == bool, 'Boolean availability required')
    for raw in ('candidate', 'control'):
        v, r = a[raw], a[raw + '_rank']
        require(v.shape == r.shape == mask.shape and np.issubdtype(v.dtype, np.floating)
                and np.issubdtype(r.dtype, np.floating), 'Float feature/rank matrices required')
        require(not np.isinf(v).any() and np.array_equal(np.isfinite(v), mask), 'Common raw availability differs')
        require(np.isfinite(r).all() and np.all(np.abs(r) <= .5), 'Finite bounded ranks required')
        bounds = (-.25, .25) if raw == 'candidate' else (0., 1.)
        require(np.all((v[mask] >= bounds[0]) & (v[mask] <= bounds[1])), 'Invalid raw range')
        require(np.array_equal(r, centered_ranks(v, mask)), 'Independent centered ranks differ')
    return a


def verify_features(panel, meta, calendar, values, audit, limit=None):
    actual = arrays(values)
    wanted, expected = independent_features(panel, meta, calendar, limit)
    for key in wanted:
        require(actual[key].shape == wanted[key].shape and
                np.array_equal(actual[key], wanted[key], equal_nan=True), 'Independent volume feature differs: ' + key)
    record_check(audit, expected)
    counters = ('events', 'symbols_per_event', 'checked_cells', 'available_cells', 'unavailable_cells',
                'event_observations', 'adjacent_pair_observations', 'rolling_medians_computed')
    return dict(passed=True, **{key: expected[key] for key in counters}, independent_scalar_rebuild=True,
        strict_prior63_median_checked=True, unsigned_volume_events_checked=True,
        adjacent_covariance_checked=True, matched_frequency_checked=True, full273_volume_support_checked=True,
        official_session_windows_checked=True, monthend_queries_checked=True, exact_common_availability=True,
        audit_checked=True, no_future_or_outcome_inputs=True, centered_high_tie_ranks_checked=True,
        predictive_model_fits=0, new_model_fits=0, computes_returns=False,
        outcome_adapter_sha256=FROZEN_HELPER_SHA)


def independent_deltas(values, dose=DOSE):
    require(type(dose) in (float, int) and dose == DOSE, 'Fixed dose .01 required')
    a = arrays(values)
    result = {arm: np.zeros(a['available'].shape, float) for arm in RAW}
    for arm, raw in RAW.items():
        for i, j in zip(*np.nonzero(a['available'])):
            if i:
                result[arm][i, j] = dose * float(a[raw + '_rank'][i, j])
    return result


def verify_deltas(values, deltas, dose=DOSE):
    require(isinstance(deltas, dict) and set(deltas) == set(RAW), 'Delta arms differ')
    expected = independent_deltas(values, dose)
    for arm, wanted in expected.items():
        actual = np.asarray(deltas[arm])
        require(actual.shape == wanted.shape and np.issubdtype(actual.dtype, np.floating)
                and np.isfinite(actual).all() and np.array_equal(actual, wanted), 'Exact delta differs: ' + arm)
        require(np.all(np.abs(actual) <= DOSE * .5), 'Unbounded delta')
    return dict(passed=True, checked_events=np.shape(values['available'])[0], dose=DOSE, max_absolute_delta=.005,
        full_candidate_universe=True, initial_delta_zero=True, common_available_only=True,
        no_eligibility_or_model_filter=True, centered_cross_sectional_ranking=True, direct_additive_pre_score=True,
        fixed_linear_mapping_checked=True,
        changed_cells={arm: int(np.count_nonzero(value)) for arm, value in expected.items()})
