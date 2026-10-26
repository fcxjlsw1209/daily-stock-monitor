"""Independent, source-only event oracle and pinned accounting for round175.

No producer feature helper or strategy engine is imported. The caller binds
source digests and the official calendar before supplying input objects. There
is no execution entry point, data loader, model fitting, or output mutation.
"""
from datetime import date
from decimal import Decimal, localcontext
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
ARMS = ('incumbent_local67', 'conditional_monthly_payoff', 'unconditional_monthly_payoff')
RAW = dict(conditional_monthly_payoff='candidate', unconditional_monthly_payoff='control')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
DOSE, SCALE = .02, .05


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_files():
    return (Path(__file__).resolve(), OUTCOME_HELPER, ACCOUNTING)


def accounting():
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('transition175_accounting', ACCOUNTING)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_outcomes(px, opening, orders, exports, results, meta):
    require(sha(OUTCOME_HELPER) == OUTCOME_SHA, 'Frozen independent outcome helper changed')
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('transition175_isolated_outcome', OUTCOME_HELPER)
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
    require(np.array_equal(np.isfinite(actual), np.isfinite(expected)), label + ': availability')
    require(np.array_equal(np.isnan(actual), np.isnan(expected)), label + ': NaNs')
    require(not np.isinf(actual).any() and not np.isinf(expected).any(), label + ': infinity')
    mask = np.isfinite(expected)
    require(np.all(np.abs(actual[mask] - expected[mask]) <= atol), label + ': numbers')


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


def schema(panel, events, calendar):
    require(isinstance(panel, dict) and set(panel) == {'adj'}, 'Panel keys differ')
    for frame in panel.values():
        require(isinstance(frame, pd.DataFrame) and isinstance(frame.index, pd.DatetimeIndex), 'Panel frame type')
        require(not frame.index.hasnans and frame.index.is_unique and frame.index.is_monotonic_increasing,
                'Invalid panel time axis')
        require(frame.index.tz is None and frame.index.equals(frame.index.normalize()), 'Daily naive dates required')
        require(frame.columns.is_unique and all(isinstance(s, str) and s for s in frame.columns), 'Panel identity')
        require(frame.index.equals(panel['adj'].index) and frame.columns.equals(panel['adj'].columns), 'Panel axes differ')
    sessions = [iso(value) for value in calendar]
    require(sessions and all(d.weekday() < 5 for d in sessions)
            and all(left < right for left, right in zip(sessions, sessions[1:])), 'Calendar order/sessions')
    lookup = {day: i for i, day in enumerate(sessions)}
    queries, width = [], 0
    for event in events:
        require(isinstance(event, (list, tuple)) and len(event) == 2, 'Event pair required')
        day, names = iso(event[0]), event[1]
        require(day in lookup and (not queries or day > queries[-1][0]), 'Event session/order')
        require(isinstance(names, (list, tuple)) and names and len(names) == len(set(names)), 'Event identities')
        require(all(isinstance(s, str) and s and s.strip() == s for s in names), 'Invalid event name')
        require(not queries or width == len(names), 'Changing event width')
        width = len(names)
        queries.append((day, list(names), lookup[day]))
    return sessions, queries, width


def empty_measure(reason):
    return dict(candidate=None, control=None, available=False, reason=reason,
        endpoint_indices=[], states=[], log_returns=[], current_state=None,
        training_labels=[], matching_transition_indices=[], matching_labels=[],
        matching_count=0, unconditional_mean=None, matching_sum=None, matching_mean=None, shrunken_payoff=None)


def measure_oracle(adjusted):
    """Full source validation, exact price-state signs, Decimal endpoint logs."""
    require(np.shape(adjusted) == (1261,), 'Exactly 1261 adjusted prices required')
    try:
        prices = [float(x) for x in adjusted]
    except (ValueError, TypeError, OverflowError):
        return empty_measure('nonnumeric_adjusted_price')
    if any(not math.isfinite(x) or x <= 0. for x in prices):
        return empty_measure('invalid_adjusted_price')
    endpoints = list(range(0, 1261, 21))
    states = [(prices[b] > prices[a]) - (prices[b] < prices[a])
              for a, b in zip(endpoints[:-1], endpoints[1:])]
    with localcontext() as context:
        context.prec = 70
        levels = [Decimal.from_float(prices[i]).ln() for i in endpoints]
        changes = [b - a for a, b in zip(levels[:-1], levels[1:])]
        labels = changes[1:]
        predecessors, query = states[:-1], states[-1]
        matched = [i for i, state in enumerate(predecessors) if state == query]
        mean = sum(labels, Decimal(0)) / Decimal(59)
        subtotal = sum((labels[i] for i in matched), Decimal(0))
        candidate = (subtotal + Decimal(20) * mean) / Decimal(len(matched) + 20)
    result = dict(candidate=None, control=None, available=False, reason='insufficient_matches',
        endpoint_indices=endpoints, states=states, log_returns=[float(x) for x in changes],
        current_state=query, training_labels=[float(x) for x in labels],
        matching_transition_indices=[i + 1 for i in matched], matching_count=len(matched),
        matching_labels=[float(labels[i]) for i in matched], unconditional_mean=float(mean),
        matching_sum=float(subtotal), matching_mean=float(subtotal / len(matched)) if matched else None,
        shrunken_payoff=None)
    if len(matched) >= 10:
        result.update(candidate=float(candidate), control=float(mean), available=True, reason='available',
                      shrunken_payoff=float(candidate))
    return result


def independent_features(panel, events, calendar, limit=None):
    sessions, queries, width = schema(panel, events, calendar)
    require(limit is None or (type(limit) is int and 0 <= limit <= len(queries)), 'Invalid prefix limit')
    queries = queries if limit is None else queries[:limit]
    values = dict(candidate=np.full((len(queries), width), np.nan),
                  control=np.full((len(queries), width), np.nan), available=np.zeros((len(queries), width), bool))
    rows, counts = [], {}
    for i, (day, names, t) in enumerate(queries):
        enough = t >= 1302
        price_days = sessions[t - 1302:t - 41] if enough else []
        missing = [str(d.date()) for d in price_days if d not in panel['adj'].index]
        cells = []
        for j, symbol in enumerate(names):
            if not enough:
                cell = empty_measure('insufficient_history')
            elif symbol not in panel['adj'].columns:
                cell = empty_measure('missing_symbol')
            elif missing:
                cell = empty_measure('missing_official_session')
            else:
                cell = measure_oracle([panel['adj'].at[d, symbol] for d in price_days])
            cell['symbol'] = symbol
            cell['endpoint_dates'] = [str(price_days[k].date()) for k in cell['endpoint_indices']]
            cell['matching_predictor_end_dates'] = [str(price_days[k * 21].date())
                for k in cell['matching_transition_indices']]
            cell['matching_label_end_dates'] = [str(price_days[(k + 1) * 21].date())
                for k in cell['matching_transition_indices']]
            if cell['available']:
                for name in ('candidate', 'control'):
                    values[name][i, j] = cell[name]
                values['available'][i, j] = True
            counts[cell['reason']] = counts.get(cell['reason'], 0) + 1
            cells.append(cell)
        rows.append(dict(date=str(day.date()), symbols=names, session_position=t,
            adjusted_start=str(price_days[0].date()) if enough else None,
            adjusted_end=str(price_days[-1].date()) if enough else None,
            missing_official_sessions=missing, cells=cells))
    audit = dict(events=len(queries), symbols_per_event=width, available_cells=int(values['available'].sum()),
        unavailable_cells=int((~values['available']).sum()), reason_counts=counts, event_history=rows,
        adjusted_count=1261, history_sessions=1260, skip_sessions=42,
        block_sessions=21, block_count=60, training_transition_count=59, minimum_matches=10,
        prior_mass=20, common_availability=True, historical_matured_price_outcomes_used=True,
        portfolio_returns_calculated=False, ranking_performed=False, trading_eligibility_modified=False,
        predictive_model_fits=0)
    return values, audit


def arrays(values):
    require(isinstance(values, dict) and set(values) == {'candidate', 'control', 'available'}, 'Value keys differ')
    a = {key: np.asarray(value) for key, value in values.items()}
    mask = a['available']
    require(mask.ndim == 2 and mask.dtype == bool, 'Boolean matrix availability required')
    for key in ('candidate', 'control'):
        require(a[key].shape == mask.shape and np.issubdtype(a[key].dtype, np.floating), 'Raw shape/dtype differs')
        require(not np.isinf(a[key]).any() and np.array_equal(np.isfinite(a[key]), mask), 'Common raw availability differs')
    return a


def verify_features(panel, events, calendar, values, audit):
    actual = arrays(values)
    wanted, expected_audit = independent_features(panel, events, calendar)
    require(np.array_equal(actual['available'], wanted['available']), 'Independent state support differs')
    for key in ('candidate', 'control'):
        close(actual[key], wanted[key], 'Independent ' + key)
    record_check(audit, expected_audit)
    return dict(passed=True, events=len(events), symbols_per_event=wanted['available'].shape[1],
        checked_cells=int(wanted['available'].size), available_cells=int(wanted['available'].sum()),
        unavailable_cells=int((~wanted['available']).sum()), independent_scalar_oracle=True,
        independent_scalar_rebuild=True, decimal_endpoint_logs=True,
        exact_common_availability=True, block_partition_checked=True,
        query_and_transition_alignment_checked=True, minimum_support_checked=True,
        event_dates_and_label_ends_checked=True, no_future_or_outcome_inputs=True, audit_checked=True,
        no_cross_sectional_ranking=True, predictive_model_fits=0, computes_returns=False,
        historical_matured_price_labels_checked=True, portfolio_returns_calculated=False,
        outcome_input_meaning='No future-of-signal prices or portfolio outcomes; matured historical transition labels explicitly rebuilt.')


def independent_deltas(values, dose=DOSE, scale=SCALE):
    require(type(dose) in (int, float) and dose == DOSE and type(scale) in (int, float) and scale == SCALE,
            'Fixed dose=.02 and scale=.05 required')
    a = arrays(values)
    result = {}
    for arm, key in RAW.items():
        delta = np.zeros(a['available'].shape)
        for i, j in zip(*np.nonzero(a['available'])):
            if i:
                delta[i, j] = dose * math.tanh(float(a[key][i, j]) / scale)
        result[arm] = delta
    return result


def verify_deltas(values, deltas, dose=DOSE, scale=SCALE):
    require(set(deltas) == set(RAW), 'Delta arms differ')
    expected = independent_deltas(values, dose, scale)
    for arm, target in expected.items():
        close(deltas[arm], target, 'Direct delta ' + arm, atol=2e-17)
        actual = np.asarray(deltas[arm])
        require(np.isfinite(actual).all() and np.all(np.abs(actual) <= dose), 'Unbounded delta')
        require(not len(actual) or np.array_equal(actual[0], np.zeros(actual.shape[1])), 'Initial delta not exactly zero')
        require(np.array_equal(actual[~np.asarray(values['available'])], np.zeros((~np.asarray(values['available'])).sum())),
                'Missing source delta not exactly zero')
    return dict(passed=True, checked_events=arrays(values)['available'].shape[0], dose=DOSE, scale=SCALE,
        full_candidate_universe=True, common_available_only=True, initial_delta_zero=True,
        no_cross_sectional_ranking=True, no_eligibility_or_model_filter=True,
        changed_cells={arm: int(np.count_nonzero(v)) for arm, v in expected.items()})
