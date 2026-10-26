"""Independent scalar verification for round182; no production feature imports.

All source selection is by explicit official-session positions. The oracle uses
the literal difference of pointwise logarithms, as registered, and never uses
prices after t-42. This module has no execution entry point or output writes.
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
ARMS = ('incumbent_local67', 'trend_gated_local', 'uniform_local_shrink')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
INCREMENT_LIMIT = .0125
TOLERANCE = 1e-12


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_files():
    return (Path(__file__).resolve(), OUTCOME_HELPER, ACCOUNTING)


def accounting():
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('trend182_accounting', ACCOUNTING)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_outcomes(px, opening, orders, exports, results, meta):
    require(sha(OUTCOME_HELPER) == OUTCOME_SHA, 'Frozen independent outcome helper changed')
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('trend182_isolated_outcome', OUTCOME_HELPER)
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
    return dict(available=False, consistent=False, reason=reason,
                block_log_returns=[], block_return_estimates=0)


def finite_sum(values):
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Invalid absolute increment sum') from exc
    if not math.isfinite(result):
        raise FloatingPointError('Nonfinite absolute increment sum')
    return result


def measure_oracle(adjusted):
    require(np.shape(adjusted) == (211,), 'Exactly 211 adjusted prices required')
    try:
        prices = [float(value) for value in adjusted]
    except (TypeError, ValueError, OverflowError):
        return empty('nonnumeric_adjusted_price')
    if any(not math.isfinite(value) or value <= 0. for value in prices):
        return empty('invalid_adjusted_price')
    try:
        logs = [math.log(prices[i]) for i in (0, 70, 140, 210)]
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Invalid pointwise log') from exc
    if not all(math.isfinite(value) for value in logs):
        raise FloatingPointError('Nonfinite pointwise log')
    returns = [last - first for first, last in zip(logs, logs[1:])]
    if not all(math.isfinite(value) for value in returns):
        raise FloatingPointError('Nonfinite block log return')
    return dict(available=True, consistent=all(value > 0. for value in returns),
                reason='available', block_log_returns=returns, block_return_estimates=3)


def independent_features(panel, meta, calendar, limit=None):
    sessions, queries, width = schema(panel, meta, calendar)
    require(limit is None or (type(limit) is int and 0 <= limit <= len(queries)), 'Invalid prefix limit')
    queries = queries if limit is None else queries[:limit]
    values = dict(consistent=np.zeros((len(queries), width), bool),
                  available=np.zeros((len(queries), width), bool))
    history, counts, estimates = [], {}, 0
    for i, (day, names, t) in enumerate(queries):
        days = sessions[t - 252:t - 41] if t >= 252 else []
        dates = [str(d.date()) for d in days]
        windows = None if not days else {
            'formation': dict(price_start=dates[0], price_end=dates[210], return_count=210),
            'blocks': [dict(start_index=k, end_index=k+70, start_date=dates[k], end_date=dates[k+70])
                       for k in (0, 70, 140)]}
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
            for key in values:
                values[key][i, j] = cell[key]
            counts[cell['reason']] = counts.get(cell['reason'], 0) + 1
            estimates += cell['block_return_estimates']
            cells.append(cell)
        history.append(dict(date=str(day.date()), symbols=names, session_position=t,
                            price_start=dates[0] if dates else None, price_end=dates[-1] if dates else None,
                            windows=windows, missing_official_sessions=missing, cells=cells))
    audit = dict(events=len(queries), symbols_per_event=width, checked_cells=int(values['available'].size),
                 price_count=211, skip_sessions=42, formation_return_count=210, block_count=3,
                 block_sessions=70, consistent_cells=int(values['consistent'].sum()),
                 available_cells=int(values['available'].sum()), unavailable_cells=int((~values['available']).sum()),
                 reason_counts=counts, block_return_estimates=estimates, predictive_model_fits=0,
                 portfolio_returns_calculated=False, ranking_performed=False,
                 trading_eligibility_modified=False, event_history=history)
    return values, audit


def arrays(values):
    require(isinstance(values, dict) and set(values) == {'consistent', 'available'}, 'Value keys differ')
    a = {key: np.asarray(value) for key, value in values.items()}
    mask = a['available']
    require(mask.ndim == 2 and mask.dtype == bool, 'Boolean matrix availability required')
    require(a['consistent'].shape == mask.shape and a['consistent'].dtype == bool, 'Boolean consistency matrix required')
    require(not np.any(a['consistent'] & ~mask), 'Unavailable cell cannot be consistent')
    return a


def verify_features(panel, meta, calendar, values, audit):
    actual = arrays(values)
    wanted, expected_audit = independent_features(panel, meta, calendar)
    require(np.array_equal(actual['available'], wanted['available']), 'Independent common support differs')
    require(np.array_equal(actual['consistent'], wanted['consistent']), 'Independent consistency differs')
    record_check(audit, expected_audit)
    return dict(passed=True, events=len(meta['dates']), symbols_per_event=wanted['available'].shape[1],
                checked_cells=int(wanted['available'].size), available_cells=int(wanted['available'].sum()),
                unavailable_cells=int((~wanted['available']).sum()), independent_scalar_rebuild=True,
                three_70_return_blocks_checked=True, strict_positive_consistency_checked=True,
                official_session_windows_checked=True,
                full_211_price_support_checked=True, exact_common_availability=True, audit_checked=True,
                no_future_or_outcome_inputs=True, no_cross_sectional_ranking=True,
                consistent_cells=expected_audit['consistent_cells'],
                block_return_estimates=expected_audit['block_return_estimates'], predictive_model_fits=0,
                computes_returns=False, portfolio_returns_calculated=False)


def routing_inputs(pre, upstream, increment, consistent, available):
    values = arrays(dict(consistent=consistent, available=available))
    shape = values['available'].shape
    p, b, d = (np.asarray(value, dtype=float) for value in (pre, upstream, increment))
    require(p.shape == b.shape == d.shape == shape, 'Routing matrices must share the prefix shape')
    require(not np.isinf(p).any() and not np.isinf(b).any(), 'Infinite original score')
    require(np.array_equal(np.isnan(p), np.isnan(b)), 'Original score missing patterns differ')
    require(np.isfinite(d).all() and (np.abs(d) <= INCREMENT_LIMIT + TOLERANCE).all(), 'Invalid frozen local increment')
    require(np.isfinite(b[d != 0.]).all(), 'Nonzero increment on missing score')
    reconstructed = b.copy()
    for i, j in zip(*np.nonzero(d)):
        reconstructed[i, j] = float(b[i, j]) + float(d[i, j])
    require(np.array_equal(reconstructed, p, equal_nan=True), 'Original pre-score reconstruction is not exact')
    return p, b, d, values['consistent'], values['available']


def same(first, second):
    return bool(first == second or (math.isnan(first) and math.isnan(second)))


def independent_routing(pre, upstream, increment, consistent, available):
    p, b, d, consistency, valid = routing_inputs(pre, upstream, increment, consistent, available)
    scores = {arm: p.copy() for arm in ARMS[1:]}
    history, fractions, candidate_removed, control_removed = [], [], [], []
    changes = {arm: 0 for arm in scores}
    for i in range(len(p)):
        denominator = finite_sum([abs(float(d[i, j])) for j in range(p.shape[1]) if valid[i, j]])
        numerator = finite_sum([abs(float(d[i, j])) for j in range(p.shape[1]) if valid[i, j] and consistency[i, j]])
        fraction = numerator / denominator if denominator > 0. and i > 0 else 0.
        require(math.isfinite(fraction) and 0. <= fraction <= 1., 'Invalid uniform removed fraction')
        removed = numerator if i > 0 else 0.
        uniform = finite_sum([fraction * abs(float(d[i, j])) for j in range(p.shape[1]) if valid[i, j]])
        require(abs(removed - uniform) <= TOLERANCE, 'Unmatched removed increment budgets')
        if i > 0:
            for j in range(p.shape[1]):
                if not valid[i, j] or d[i, j] == 0.:
                    continue
                if consistency[i, j]:
                    scores[ARMS[1]][i, j] = b[i, j]
                if fraction == 1.:
                    scores[ARMS[2]][i, j] = b[i, j]
                elif fraction != 0.:
                    value = float(b[i, j]) + (1. - fraction) * float(d[i, j])
                    if not math.isfinite(value):
                        raise FloatingPointError('Nonfinite routed score')
                    scores[ARMS[2]][i, j] = value
        row_changes = {arm: sum(not same(scores[arm][i, j], p[i, j]) for j in range(p.shape[1])) for arm in scores}
        for arm in scores:
            changes[arm] += row_changes[arm]
        fractions.append(fraction); candidate_removed.append(removed); control_removed.append(uniform)
        history.append(dict(row=i, initial_neutral=i == 0, valid_cells=int(valid[i].sum()),
                            consistent_valid_cells=int((consistency[i] & valid[i]).sum()),
                            available_abs_increment=denominator, consistent_abs_increment=numerator,
                            uniform_removed_fraction=fraction, candidate_removed_abs_increment=removed,
                            control_removed_abs_increment=uniform, matched_budget_error=abs(removed - uniform),
                            changed_cells=row_changes))
    audit = dict(events=len(p), symbols_per_event=p.shape[1], checked_cells=p.size,
                 increment_abs_limit=INCREMENT_LIMIT, tolerance=TOLERANCE,
                 original_pre_reconstruction_exact=True, initial_scores_unchanged=True,
                 unchanged_cell_copy_exact=True, matched_removed_budgets=True,
                 no_cross_sectional_ranking=True, predictive_model_fits=0, portfolio_returns_calculated=False,
                 episode_applications=0, uniform_removed_fraction=fractions,
                 candidate_removed_abs_increment=candidate_removed, control_removed_abs_increment=control_removed,
                 changed_cells=changes, event_history=history)
    return scores, audit


def verify_routing(pre, upstream, increment, consistent, available, scores, audit):
    require(isinstance(scores, dict) and set(scores) == set(ARMS[1:]), 'Routed score arms differ')
    expected, expected_audit = independent_routing(pre, upstream, increment, consistent, available)
    p, b, d, c, valid = routing_inputs(pre, upstream, increment, consistent, available)
    for arm, wanted in expected.items():
        close(scores[arm], wanted, 'Routed score ' + arm)
        actual = np.asarray(scores[arm], float)
        unchanged = ~valid | (d == 0.)
        if len(p):
            unchanged[0] = True
        if arm == ARMS[1]:
            unchanged |= ~c
            removed = valid & c & (d != 0.)
        else:
            fractions = np.asarray(expected_audit['uniform_removed_fraction'])[:, None]
            unchanged |= fractions == 0.
            removed = valid & (fractions == 1.) & (d != 0.)
        if len(p):
            removed[0] = False
        require(np.array_equal(actual[unchanged], p[unchanged], equal_nan=True), 'Unchanged scores were not copied exactly')
        require(np.array_equal(actual[removed], b[removed], equal_nan=True), 'Full removal did not copy upstream exactly')
    record_check(audit, expected_audit)
    return dict(passed=True, events=len(p), symbols_per_event=p.shape[1], checked_cells=p.size,
                independent_scalar_rebuild=True, original_increment_identity_checked=True,
                exact_unchanged_cells_checked=True, matched_absolute_removal_checked=True,
                original_pre_reconstruction_exact=True, initial_scores_unchanged=True,
                unchanged_cell_copy_exact=True, matched_removed_budgets=True,
                no_cross_sectional_ranking=True, no_new_eligibility_filter=True,
                predictive_model_fits=0, portfolio_returns_calculated=False, episode_applications=0,
                changed_cells=expected_audit['changed_cells'],
                uniform_removed_fraction=expected_audit['uniform_removed_fraction'])
