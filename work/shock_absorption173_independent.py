"""Independent, source-only event oracle and pinned accounting for round173.

No producer feature helper or strategy engine is imported. The caller binds
source digests and the official calendar before supplying input objects. There
is no execution entry point, data loader, model fitting, or output mutation.
"""
from datetime import date
from fractions import Fraction
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
ARMS = ('incumbent_local67', 'activity_recovery_contrast', 'ordinary_decline_recovery')
RAW = dict(activity_recovery_contrast='candidate', ordinary_decline_recovery='control')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
DOSE, SCALE = .01, .05


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_files():
    return (Path(__file__).resolve(), OUTCOME_HELPER, ACCOUNTING)


def accounting():
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('shock173_accounting', ACCOUNTING)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_outcomes(px, opening, orders, exports, results, meta):
    require(sha(OUTCOME_HELPER) == OUTCOME_SHA, 'Frozen independent outcome helper changed')
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('shock173_isolated_outcome', OUTCOME_HELPER)
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
    require(isinstance(panel, dict) and set(panel) == {'adj', 'close', 'volume'}, 'Panel keys differ')
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
        selected_anchors=[], high_anchors=[], low_anchors=[], invalid_activity_anchors=[],
        n_all=0, n_high=0, n_low=0, labels=[], classifications=[])


def measure_oracle(adjusted, stored_close, stored_volume):
    """Scalar Fraction activity threshold and log-level outcome differences."""
    arrays = []
    for vector, count, name in ((adjusted, 1261, 'adjusted_price'),
                               (stored_close, 1259, 'stored_close'),
                               (stored_volume, 1259, 'stored_volume')):
        require(np.shape(vector) == (count,), 'Exact 1261/1259/1259 input counts required')
        try:
            array = [float(value) for value in vector]
        except (ValueError, TypeError, OverflowError):
            return empty_measure('nonnumeric_' + name)
        arrays.append(array)
    for array, name in zip(arrays, ('adjusted_price', 'stored_close', 'stored_volume')):
        if any(not math.isfinite(x) or (x < 0 if name == 'stored_volume' else x <= 0) for x in array):
            return empty_measure('invalid_' + name)
    prices, close_values, volume = arrays
    result = empty_measure('available')
    # Selection is completed without inspecting activity or forward labels.
    anchors = []
    next_allowed = 1
    for anchor in range(1, 1240):
        if anchor >= next_allowed and prices[anchor] < prices[anchor - 1]:
            anchors.append(anchor)
            next_allowed = anchor + 21
    activity = [Fraction.from_float(c) * Fraction.from_float(v) for c, v in zip(close_values, volume)]
    labels = [math.log(prices[s + 21]) - math.log(prices[s]) for s in anchors]
    result.update(selected_anchors=anchors, labels=labels, n_all=len(anchors))
    for s in anchors:
        index = s + 19
        prior = sum(activity[index - 20:index], Fraction(0))
        if prior == 0:
            result['invalid_activity_anchors'].append(s)
            result['classifications'].append(None)
        elif 20 * activity[index] > prior:
            result['high_anchors'].append(s)
            result['classifications'].append('H')
        else:
            result['low_anchors'].append(s)
            result['classifications'].append('L')
    high, low = result['high_anchors'], result['low_anchors']
    result.update(n_high=len(high), n_low=len(low))
    if result['invalid_activity_anchors']:
        result['reason'] = 'zero_prior_activity'
    elif len(high) < 3 or len(low) < 3:
        result['reason'] = 'insufficient_groups'
    else:
        h = math.fsum(y for y, tag in zip(labels, result['classifications']) if tag == 'H')
        l = math.fsum(y for y, tag in zip(labels, result['classifications']) if tag == 'L')
        result.update(candidate=h / (len(high) + 20) - l / (len(low) + 20),
                      control=math.fsum(labels) / (len(anchors) + 20), available=True)
    return result


def independent_features(panel, events, calendar, limit=None):
    sessions, queries, width = schema(panel, events, calendar)
    require(limit is None or (type(limit) is int and 0 <= limit <= len(queries)), 'Invalid prefix limit')
    queries = queries if limit is None else queries[:limit]
    values = dict(candidate=np.full((len(queries), width), np.nan),
                  control=np.full((len(queries), width), np.nan), available=np.zeros((len(queries), width), bool))
    rows, counts = [], {}
    for i, (day, names, t) in enumerate(queries):
        enough = t >= 1321
        price_days = sessions[t - 1302:t - 41] if enough else []
        activity_days = sessions[t - 1321:t - 62] if enough else []
        required = sorted(set(price_days + activity_days))
        missing = [str(d.date()) for d in required if d not in panel['adj'].index]
        cells = []
        for j, symbol in enumerate(names):
            if not enough:
                cell = empty_measure('insufficient_history')
            elif symbol not in panel['adj'].columns:
                cell = empty_measure('missing_symbol')
            elif missing:
                cell = empty_measure('missing_official_session')
            else:
                cell = measure_oracle([panel['adj'].at[d, symbol] for d in price_days],
                    [panel['close'].at[d, symbol] for d in activity_days],
                    [panel['volume'].at[d, symbol] for d in activity_days])
            cell['symbol'] = symbol
            for key, target in (('selected_anchors', 'selected_anchor_dates'), ('high_anchors', 'high_anchor_dates'),
                                ('low_anchors', 'low_anchor_dates'), ('invalid_activity_anchors', 'invalid_activity_anchor_dates')):
                cell[target] = [str(price_days[s].date()) for s in cell[key]]
            cell['label_end_dates'] = [str(price_days[s + 21].date()) for s in cell['selected_anchors']]
            if cell['available']:
                for name in ('candidate', 'control'):
                    values[name][i, j] = cell[name]
                values['available'][i, j] = True
            counts[cell['reason']] = counts.get(cell['reason'], 0) + 1
            cells.append(cell)
        rows.append(dict(date=str(day.date()), symbols=names, session_position=t,
            adjusted_start=str(price_days[0].date()) if enough else None,
            adjusted_end=str(price_days[-1].date()) if enough else None,
            activity_start=str(activity_days[0].date()) if enough else None,
            activity_end=str(activity_days[-1].date()) if enough else None,
            missing_official_sessions=missing, cells=cells))
    audit = dict(events=len(queries), symbols_per_event=width, available_cells=int(values['available'].sum()),
        unavailable_cells=int((~values['available']).sum()), reason_counts=counts, event_history=rows,
        adjusted_count=1261, activity_count=1259, history_sessions=1260, skip_sessions=42,
        horizon_sessions=21, cooldown_sessions=21, activity_baseline_sessions=20, prior_mass=20,
        minimum_group_anchors=3, common_availability=True, historical_matured_price_outcomes_used=True,
        portfolio_returns_calculated=False, ranking_performed=False, trading_eligibility_modified=False)
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
    require(np.array_equal(actual['available'], wanted['available']), 'Independent event support differs')
    for key in ('candidate', 'control'):
        close(actual[key], wanted[key], 'Independent ' + key)
    record_check(audit, expected_audit)
    return dict(passed=True, events=len(events), symbols_per_event=wanted['available'].shape[1],
        checked_cells=int(wanted['available'].size), available_cells=int(wanted['available'].sum()),
        unavailable_cells=int((~wanted['available']).sum()), independent_scalar_oracle=True,
        independent_scalar_rebuild=True,
        exact_activity_classification=True, exact_common_availability=True, common_cooldown_checked=True,
        common_event_cooldown_checked=True,
        event_dates_and_label_ends_checked=True, no_future_or_outcome_inputs=True, audit_checked=True,
        no_cross_sectional_ranking=True, computes_returns=False,
        historical_matured_price_labels_checked=True, portfolio_returns_calculated=False,
        outcome_input_meaning='No future-of-signal prices or portfolio outcomes; matured historical labels explicitly rebuilt.')


def independent_deltas(values, dose=DOSE, scale=SCALE):
    require(type(dose) in (int, float) and dose == DOSE and type(scale) in (int, float) and scale == SCALE,
            'Fixed dose=.01 and scale=.05 required')
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
