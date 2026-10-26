"""Independent scalar verification for round181; no production feature imports.

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
ARMS = ('incumbent_local67', 'upside_retention', 'binary_upside_retention')
RAW = dict(upside_retention='candidate', binary_upside_retention='control')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
DOSE = .05


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_files():
    return (Path(__file__).resolve(), OUTCOME_HELPER, ACCOUNTING)


def accounting():
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('retention181_accounting', ACCOUNTING)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_outcomes(px, opening, orders, exports, results, meta):
    require(sha(OUTCOME_HELPER) == OUTCOME_SHA, 'Frozen independent outcome helper changed')
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('retention181_isolated_outcome', OUTCOME_HELPER)
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
    return dict(available=False, reason=reason, candidate=None, control=None,
                retention_share_estimates=0, positive_anchor_count=0,
                anchor_indices=[], positive_log_returns=[], retained_log_moves=[],
                capped_retained_log_moves=[], binary_recovered=[],
                positive_return_sum=None, retained_return_sum=None, binary_weighted_sum=None,
                retention_share=None, binary_retention_share=None)


def finite_sum(values):
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Invalid retained-log sum') from exc
    if not math.isfinite(result):
        raise FloatingPointError('Nonfinite retained-log sum')
    return result


def measure_oracle(adjusted):
    require(np.shape(adjusted) == (253,), 'Exactly 253 adjusted prices required')
    try:
        prices = [float(value) for value in adjusted]
    except (TypeError, ValueError, OverflowError):
        return empty('nonnumeric_adjusted_price')
    if any(not math.isfinite(value) or value <= 0. for value in prices):
        return empty('invalid_adjusted_price')
    try:
        logs = [math.log(value) for value in prices]
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Invalid pointwise log') from exc
    if not all(math.isfinite(value) for value in logs):
        raise FloatingPointError('Nonfinite pointwise log')
    result = empty('no_positive_anchors')
    result['retention_share_estimates'] = 2
    weighted = []
    for i in range(1, 232):
        move = logs[i] - logs[i - 1]
        if not math.isfinite(move):
            raise FloatingPointError('Nonfinite anchor log return')
        if move <= 0.:
            continue
        retained = logs[i + 21] - logs[i - 1]
        if not math.isfinite(retained):
            raise FloatingPointError('Nonfinite matured log return')
        bounded = min(move, max(0., retained))
        recovered = prices[i + 21] >= prices[i]
        result['anchor_indices'].append(i)
        result['positive_log_returns'].append(move)
        result['retained_log_moves'].append(retained)
        result['capped_retained_log_moves'].append(bounded)
        result['binary_recovered'].append(bool(recovered))
        weighted.append(move if recovered else 0.)
    denominator = finite_sum(result['positive_log_returns'])
    retained_total = finite_sum(result['capped_retained_log_moves'])
    binary_total = finite_sum(weighted)
    result.update(positive_return_sum=denominator, retained_return_sum=retained_total,
                  binary_weighted_sum=binary_total, positive_anchor_count=len(result['anchor_indices']))
    if not result['positive_anchor_count']:
        return result
    if denominator <= 0.:
        raise FloatingPointError('Nonpositive sum of positive anchor returns')
    q, c = retained_total / denominator, binary_total / denominator
    if not all(math.isfinite(value) and 0. <= value <= 1. for value in (q, c)):
        raise FloatingPointError('Retention share outside [0,1]')
    result.update(available=True, reason='available', retention_share=q, binary_retention_share=c,
                  candidate=q - .5, control=c - .5)
    return result


def independent_features(panel, meta, calendar, limit=None):
    sessions, queries, width = schema(panel, meta, calendar)
    require(limit is None or (type(limit) is int and 0 <= limit <= len(queries)), 'Invalid prefix limit')
    queries = queries if limit is None else queries[:limit]
    values = dict(candidate=np.full((len(queries), width), np.nan),
                  control=np.full((len(queries), width), np.nan), available=np.zeros((len(queries), width), bool))
    history, counts, estimates, positive_count = [], {}, 0, 0
    for i, (day, names, t) in enumerate(queries):
        days = sessions[t - 294:t - 41] if t >= 294 else []
        dates = [str(d.date()) for d in days]
        windows = None if not days else {
            'formation': dict(price_start=dates[0], price_end=dates[252], return_count=252),
            'anchors': dict(first_date=dates[1], last_date=dates[231], count=231),
            'maturities': dict(first_date=dates[22], last_date=dates[252], horizon_sessions=21)}
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
            estimates += cell['retention_share_estimates']
            positive_count += cell['positive_anchor_count']
            cells.append(cell)
        history.append(dict(date=str(day.date()), symbols=names, session_position=t,
                            price_start=dates[0] if dates else None, price_end=dates[-1] if dates else None,
                            windows=windows, missing_official_sessions=missing, cells=cells))
    audit = dict(events=len(queries), symbols_per_event=width, checked_cells=int(values['available'].size),
                 price_count=253, skip_sessions=42, formation_return_count=252, anchor_count=231,
                 horizon_sessions=21, positive_anchor_count=positive_count,
                 available_cells=int(values['available'].sum()), unavailable_cells=int((~values['available']).sum()),
                 reason_counts=counts, retention_share_estimates=estimates, predictive_model_fits=0,
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
        require(np.all(a[key][mask] >= -.5) and np.all(a[key][mask] <= .5), 'Raw value outside fixed bounds')
    return a


def verify_features(panel, meta, calendar, values, audit):
    actual = arrays(values)
    wanted, expected_audit = independent_features(panel, meta, calendar)
    require(np.array_equal(actual['available'], wanted['available']), 'Independent common support differs')
    for key in ('candidate', 'control'):
        close(actual[key], wanted[key], 'Independent ' + key)
    record_check(audit, expected_audit)
    return dict(passed=True, events=len(meta['dates']), symbols_per_event=wanted['available'].shape[1],
                checked_cells=int(wanted['available'].size), available_cells=int(wanted['available'].sum()),
                unavailable_cells=int((~wanted['available']).sum()), independent_scalar_rebuild=True,
                two_retention_shares_checked=True, positive_log_return_anchors_checked=True,
                matured_21_session_labels_checked=True, official_session_windows_checked=True,
                full_253_price_support_checked=True, exact_common_availability=True, audit_checked=True,
                no_future_or_outcome_inputs=True, no_cross_sectional_ranking=True,
                positive_anchor_count=expected_audit['positive_anchor_count'],
                retention_share_estimates=expected_audit['retention_share_estimates'], predictive_model_fits=0,
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
        require(np.isfinite(actual).all() and np.all(actual >= -.5 * dose) and
                np.all(actual <= .5 * dose), 'Unbounded delta')
        require(not len(actual) or np.array_equal(actual[0], np.zeros(actual.shape[1])), 'Initial delta not exactly zero')
        missing = ~np.asarray(values['available'])
        require(np.array_equal(actual[missing], np.zeros(missing.sum())), 'Missing source delta not exactly zero')
    return dict(passed=True, checked_events=arrays(values)['available'].shape[0], dose=DOSE,
                full_candidate_universe=True, common_available_only=True, common_availability=True,
                initial_delta_zero=True, direct_additive_pre_score=True, no_secondary_rank=True,
                no_cross_sectional_ranking=True, no_eligibility_or_model_filter=True, direct_linear_mapping=True,
                max_absolute_delta=.025,
                changed_cells={arm: int(np.count_nonzero(v)) for arm, v in expected.items()})
