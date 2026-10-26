"""Independent scalar source audit for round179, plus frozen outcome accounting.

This module does not import a production kernel, host, or strategy engine.  It
has no entry point and performs no source loading, model fitting, or writes.
The caller is responsible for binding raw-source and official-calendar hashes.
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
ARMS = ('incumbent_local67', 'historical_downside_mix', 'neutral_downside_mix')
RAW = dict(historical_downside_mix='candidate', neutral_downside_mix='control')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
DOSE = .05
FLOOR = 1e-24


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_files():
    return (Path(__file__).resolve(), OUTCOME_HELPER, ACCOUNTING)


def accounting():
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('downside179_accounting', ACCOUNTING)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_outcomes(px, opening, orders, exports, results, meta):
    require(sha(OUTCOME_HELPER) == OUTCOME_SHA, 'Frozen independent outcome helper changed')
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('downside179_isolated_outcome', OUTCOME_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
    mask = np.isfinite(expected)
    require(np.all(np.abs(actual[mask] - expected[mask]) <= atol), label + ': values')


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
        require(type(actual) in (float, int), label + ': scalar')
        if 'sum_squares_' in label:
            require(math.isfinite(actual) and abs(actual - expected) <= 1e-38 + 1e-12 * abs(expected),
                    label + ': squared-return sum differs')
        else:
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
    frame = panel['adj']
    require(isinstance(frame, pd.DataFrame) and isinstance(frame.index, pd.DatetimeIndex), 'Panel frame type')
    require(not frame.index.hasnans and frame.index.is_unique and frame.index.is_monotonic_increasing,
            'Invalid panel time axis')
    require(frame.index.tz is None and frame.index.equals(frame.index.normalize()), 'Daily naive dates required')
    require(frame.columns.is_unique and all(isinstance(s, str) and s and s.strip() == s for s in frame.columns),
            'Panel identities')
    require(isinstance(meta, dict) and all(k in meta for k in ('dates', 'symbols', 'etfs')), 'Metadata fields')
    require(len(meta['dates']) == len(meta['symbols']), 'Event metadata lengths')
    if 'etfs' in meta:
        require(isinstance(meta['etfs'], (list, tuple, set, frozenset)) and
                len(meta['etfs']) == len(set(meta['etfs'])) and
                all(isinstance(s, str) and s and s.strip() == s for s in meta['etfs']), 'ETF metadata')
    sessions = [iso(value) for value in calendar]
    require(sessions and all(d.weekday() < 5 for d in sessions) and
            all(a < b for a, b in zip(sessions, sessions[1:])), 'Calendar order/sessions')
    lookup = {day: i for i, day in enumerate(sessions)}
    queries, width = [], 0
    for raw, names in zip(meta['dates'], meta['symbols']):
        day = iso(raw)
        require(day in lookup and (not queries or day > queries[-1][0]), 'Event session/order')
        require(isinstance(names, (list, tuple)) and names and len(names) == len(set(names)), 'Event identities')
        require(all(isinstance(s, str) and s and s.strip() == s for s in names), 'Invalid event name')
        require(not queries or width == len(names), 'Changing event width')
        width = len(names)
        queries.append((day, list(names), lookup[day]))
    return sessions, queries, width


def empty(reason):
    return dict(available=False, reason=reason,
                sum_squares_old252=None, downside_sum_squares_old252=None, share_old252=None,
                sum_squares_current63=None, downside_sum_squares_current63=None, share_current63=None,
                candidate=None, control=None, downside_share_estimates=0)


def squared_returns(prices):
    """Uncentered simple-return squares; zero returns are not negative."""
    all_squares, negative_squares = [], []
    for first, last in zip(prices, prices[1:]):
        ratio = last / first
        if not math.isfinite(ratio) or ratio <= 0.:
            raise FloatingPointError('Nonfinite or nonpositive simple price ratio')
        value = ratio - 1.
        square = value * value
        if not math.isfinite(square):
            raise FloatingPointError('Nonfinite simple-return square')
        all_squares.append(square)
        if value < 0.:
            negative_squares.append(square)
    try:
        total = math.fsum(all_squares)
        negative = math.fsum(negative_squares)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Nonfinite simple-return square sum') from exc
    if not math.isfinite(total) or not math.isfinite(negative):
        raise FloatingPointError('Nonfinite simple-return square sum')
    return total, negative


def measure_oracle(adjusted):
    require(np.shape(adjusted) == (295,), 'Exactly 295 adjusted prices required')
    try:
        prices = [float(value) for value in adjusted]
    except (TypeError, ValueError, OverflowError):
        return empty('nonnumeric_adjusted_price')
    if any(not math.isfinite(value) or value <= 0. for value in prices):
        return empty('invalid_adjusted_price')
    old, recent = squared_returns(prices[:253]), squared_returns(prices[231:])
    result = empty('downside_denominator_floor')
    result.update(sum_squares_old252=old[0], downside_sum_squares_old252=old[1],
                  sum_squares_current63=recent[0], downside_sum_squares_current63=recent[1],
                  downside_share_estimates=2)
    if old[0] <= FLOOR or recent[0] <= FLOOR:
        return result
    old_share, recent_share = old[1] / old[0], recent[1] / recent[0]
    candidate, control = old_share - recent_share, .5 - recent_share
    if not all(math.isfinite(value) for value in (old_share, recent_share, candidate, control)):
        raise FloatingPointError('Nonfinite downside-share score')
    if not (0. <= old_share <= 1. and 0. <= recent_share <= 1.):
        raise FloatingPointError('Downside share outside [0,1]')
    result.update(available=True, reason='available', share_old252=old_share,
                  share_current63=recent_share, candidate=candidate, control=control)
    return result


def independent_features(panel, meta, calendar, limit=None):
    sessions, queries, width = schema(panel, meta, calendar)
    require(limit is None or (type(limit) is int and 0 <= limit <= len(queries)), 'Invalid prefix limit')
    queries = queries if limit is None else queries[:limit]
    values = dict(candidate=np.full((len(queries), width), np.nan),
                  control=np.full((len(queries), width), np.nan), available=np.zeros((len(queries), width), bool))
    history, counts, estimates = [], {}, 0
    for i, (day, names, t) in enumerate(queries):
        days = sessions[t - 294:t + 1] if t >= 294 else []
        dates = [str(d.date()) for d in days]
        windows = None if not days else {
            'old252': dict(price_start=dates[0], price_end=dates[252], return_count=252),
            'current63': dict(price_start=dates[231], price_end=dates[294], return_count=63)}
        missing = [str(d.date()) for d in days if d not in panel['adj'].index]
        cells = []
        for j, symbol in enumerate(names):
            if not days:
                cell = empty('insufficient_history')
            elif symbol not in panel['adj'].columns:
                cell = empty('missing_symbol')
            elif missing:
                cell = empty('missing_official_session')
            else:
                cell = measure_oracle([panel['adj'].at[d, symbol] for d in days])
            cell.update(symbol=symbol, date=str(day.date()), windows=windows)
            if cell['available']:
                for key in ('candidate', 'control'):
                    values[key][i, j] = cell[key]
                values['available'][i, j] = True
            counts[cell['reason']] = counts.get(cell['reason'], 0) + 1
            estimates += cell['downside_share_estimates']
            cells.append(cell)
        history.append(dict(date=str(day.date()), symbols=names, session_position=t,
                            price_start=dates[0] if dates else None, price_end=dates[-1] if dates else None,
                            windows=windows, missing_official_sessions=missing, cells=cells))
    audit = dict(events=len(queries), symbols_per_event=width, checked_cells=int(values['available'].size),
                 price_count=295, skip_sessions=42, old_return_count=252, current_return_count=63,
                 denominator_floor=FLOOR,
                 available_cells=int(values['available'].sum()), unavailable_cells=int((~values['available']).sum()),
                 reason_counts=counts, downside_share_estimates=estimates, predictive_model_fits=0,
                 common_availability=True, portfolio_returns_calculated=False, ranking_performed=False,
                 trading_eligibility_modified=False, event_history=history)
    return values, audit


def arrays(values):
    require(isinstance(values, dict) and set(values) == {'candidate', 'control', 'available'}, 'Value keys differ')
    a = {key: np.asarray(value) for key, value in values.items()}
    mask = a['available']
    require(mask.ndim == 2 and mask.dtype == bool, 'Boolean matrix availability required')
    for key in ('candidate', 'control'):
        require(a[key].shape == mask.shape and np.issubdtype(a[key].dtype, np.floating), 'Raw shape/dtype differs')
        require(not np.isinf(a[key]).any() and np.array_equal(np.isfinite(a[key]), mask), 'Common raw availability differs')
        bound = 1. if key == 'candidate' else .5
        require(np.all(a[key][mask] >= -bound) and np.all(a[key][mask] <= bound), 'Raw value outside fixed bounds')
    return a


def verify_features(panel, meta, calendar, values, audit):
    actual = arrays(values)
    wanted, expected_audit = independent_features(panel, meta, calendar)
    require(np.array_equal(actual['available'], wanted['available']), 'Independent common support differs')
    for key in ('candidate', 'control'):
        close(actual[key], wanted[key], 'Independent ' + key)
    require(audit.get('denominator_floor') == FLOOR, 'Fixed denominator floor differs')
    record_check(audit, expected_audit)
    return dict(passed=True, events=len(meta['dates']), symbols_per_event=wanted['available'].shape[1],
                checked_cells=int(wanted['available'].size), available_cells=int(wanted['available'].sum()),
                unavailable_cells=int((~wanted['available']).sum()), independent_scalar_rebuild=True,
                two_downside_shares_checked=True, strict_negative_simple_returns_checked=True,
                official_session_windows_checked=True,
                full_295_price_support_checked=True, exact_common_availability=True, audit_checked=True,
                no_future_or_outcome_inputs=True, no_cross_sectional_ranking=True,
                downside_share_estimates=expected_audit['downside_share_estimates'], predictive_model_fits=0,
                computes_returns=False, portfolio_returns_calculated=False)


def independent_deltas(values, dose=DOSE):
    require(type(dose) in (int, float) and dose == DOSE, 'Fixed dose=.05 required')
    a = arrays(values)
    result = {}
    for arm, key in RAW.items():
        delta = np.zeros(a['available'].shape)
        for i, j in zip(*np.nonzero(a['available'])):
            if i:
                delta[i, j] = dose * float(a[key][i, j])
        result[arm] = delta
    return result


def verify_deltas(values, deltas, dose=DOSE):
    require(isinstance(deltas, dict) and set(deltas) == set(RAW), 'Delta arms differ')
    expected = independent_deltas(values, dose)
    for arm, target in expected.items():
        close(deltas[arm], target, 'Direct delta ' + arm, atol=2e-17)
        actual = np.asarray(deltas[arm])
        bound = dose if RAW[arm] == 'candidate' else .5 * dose
        require(np.isfinite(actual).all() and np.all(actual >= -bound) and np.all(actual <= bound), 'Unbounded delta')
        require(not len(actual) or np.array_equal(actual[0], np.zeros(actual.shape[1])), 'Initial delta not exactly zero')
        missing = ~np.asarray(values['available'])
        require(np.array_equal(actual[missing], np.zeros(missing.sum())), 'Missing source delta not exactly zero')
    return dict(passed=True, checked_events=arrays(values)['available'].shape[0], dose=DOSE,
                full_candidate_universe=True, common_available_only=True, common_availability=True,
                initial_delta_zero=True, direct_additive_pre_score=True, no_secondary_rank=True,
                no_cross_sectional_ranking=True, no_eligibility_or_model_filter=True, direct_linear_mapping=True,
                changed_cells={arm: int(np.count_nonzero(v)) for arm, v in expected.items()})
