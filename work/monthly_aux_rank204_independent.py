"""Independent cached auxiliary-rank oracle; no market-data or model entry point.

Scalar comparisons rebuild the full candidate-domain average-tie ranks.
All transform/audit comparisons are exact binary64 comparisons. The 1e-12
support bound is validation only: no clipping, tie tolerance, or rounding.
"""
import hashlib
import importlib.util
import math
import numbers
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FROZEN_HELPER = ROOT / 'work/monthly_sector_recurrence192_independent.py'
FROZEN_HELPER_SHA = 'f78cc005827fd977ec431bdec4af0f4b7160694cb3d382df0115e8e00e0bedf8'
ARMS = ('incumbent_local67', 'aux_rank_full', 'aux_affine_control')
DELTA_KEYS = {'aux_rank_full': 'candidate_delta', 'aux_affine_control': 'control_delta'}
VALUE_KEYS = ('support', 'weight', 'candidate', 'control', 'candidate_delta', 'control_delta')
FEATURE_FLAGS = ('passed', 'independent_scalar_rebuild', 'full_auxiliary_support_checked',
    'core_weights_checked', 'full_candidate_universe', 'average_tie_ranks_checked',
    'affine_span_control_checked', 'two_level_equivalence_checked', 'constant_rows_unchanged',
    'first_partial_unchanged', 'effective_support_weight_checked', 'no_eligibility_rank_filter',
    'no_new_availability_filter', 'no_future_or_outcome_inputs', 'audit_checked')
DELTA_FLAGS = ('passed', 'independent_delta_rebuild', 'effective_support_weight_checked',
    'full_auxiliary_replacement_checked', 'first_partial_unchanged', 'constant_rows_unchanged',
    'core_weights_checked', 'no_eligibility_delta_filter', 'source_values_checked')
COUNT_FIELDS = ('events', 'symbols_per_event', 'checked_cells', 'rank_rows',
                'constant_rows', 'two_level_rows', 'active_forecast_cells')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def frozen_helper():
    require(hashlib.sha256(FROZEN_HELPER.read_bytes()).hexdigest() == FROZEN_HELPER_SHA,
            'Frozen192 independent accounting helper changed')
    spec = importlib.util.spec_from_file_location('aux204_frozen_accounting', FROZEN_HELPER)
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
    result = helper.audit_outcomes(px, opening, orders, exports, results, meta)
    require(result['checked_paths'] == 9 and result['checked_slices'] == 45 and
            set(result['paths']) == set(ARMS), 'Incomplete nine-path accounting')
    result['outcome_adapter_sha256'] = FROZEN_HELPER_SHA
    return result


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
        require(type(actual) is type(expected) and actual == expected, label + ': exact value/type')


def finite(value):
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite cached-transform arithmetic')
    return value


def _numeric(raw, name, finite_required=True):
    require(all(isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_))
                for x in raw.flat), name + ': real nonboolean numeric values required')
    try:
        result = raw.astype(np.float64, copy=True)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(name + ': binary64 conversion failed') from exc
    if finite_required:
        require(np.isfinite(result).all(), name + ': finite values required')
    return result


def _bool(raw, name):
    require(all(isinstance(x, (bool, np.bool_)) for x in raw.flat),
            name + ': exact boolean values required')
    return raw.astype(bool, copy=True)


def source_prefix(arrays, limit=None):
    require(isinstance(arrays, dict), 'Original arrays dictionary required')
    keys = ('baseline', 'core', 'core_weight', 'state_calm10', 'state_weak_narrow',
            'extension_available', 'forecast_available', 'eligible', 'rank_log_excess')
    require(all(key in arrays for key in keys), 'Missing original cached input')
    source = {key: np.asarray(arrays[key]) for key in keys}
    shape = source['baseline'].shape
    require(len(shape) == 2 and shape[1] >= 1, 'Nonempty candidate width required')
    full, width = shape
    require(limit is None or type(limit) is int and 0 <= limit <= full, 'Invalid prefix')
    n = full if limit is None else limit
    require(source['core'].shape == shape, 'Core shape differs')
    require(source['core_weight'].shape in ((full,), (full, 1), shape), 'Core weight shape differs')
    for key in ('state_calm10', 'state_weak_narrow'):
        require(source[key].shape == (full,), 'State shape differs: ' + key)
    for key in ('extension_available', 'forecast_available', 'eligible', 'rank_log_excess'):
        require(source[key].shape == shape, 'Matrix shape differs: ' + key)
    # Shapes may be inspected globally; values are consumed only after slicing.
    out = {}
    for key in ('baseline', 'core', 'rank_log_excess'):
        out[key] = _numeric(source[key][:n], key, key != 'rank_log_excess')
    for key in ('state_calm10', 'state_weak_narrow', 'extension_available',
                'forecast_available', 'eligible'):
        out[key] = _bool(source[key][:n], key)
    cw = _numeric(source['core_weight'][:n], 'core_weight')
    if cw.ndim == 1:
        cw = cw[:, None]
    out['core_weight'] = np.broadcast_to(cw, (n, width)).copy()
    for i in range(n):
        required = .50 if out['state_calm10'][i] else .52
        require(all(float(x) == required for x in out['core_weight'][i]), 'Core weight/state mismatch')
    return out, n, width


def row_targets(support):
    """Full-domain pair-count ranks, not producer sort/rank operations."""
    s = [float(x) for x in support]
    require(s and all(math.isfinite(x) for x in s), 'Finite support row required')
    width = len(s)
    levels = len(set(s))
    if levels == 1:
        return [0.] * width, [0.] * width, levels
    q = []
    for x in s:
        lower = sum(y < x for y in s)
        equal = sum(y == x for y in s)
        q.append((lower + (equal - 1) / 2.) / (width - 1) - .5)
    if levels == 2:
        return q, q.copy(), levels
    smin, smax, qmin, qmax = min(s), max(s), min(q), max(q)
    control = []
    for x in s:
        if x == smin:
            value = qmin
        elif x == smax:
            value = qmax
        else:
            value = finite(qmin + finite(finite(x - smin) * finite(qmax - qmin)) / finite(smax - smin))
        control.append(value)
    return q, control, levels


def independent_features(arrays, limit=None):
    source, n, width = source_prefix(arrays, limit)
    values = {key: np.empty((n, width), dtype=np.float64) for key in VALUE_KEYS}
    rows = []
    for i in range(n):
        active_count = 0
        for j in range(width):
            cw = float(source['core_weight'][i, j])
            support = finite(finite(float(source['baseline'][i, j]) - float(source['core'][i, j])) / (1. - cw))
            require(abs(support) <= .5 + 1e-12, 'Original support exceeds validation bound')
            calm, weak = bool(source['state_calm10'][i]), bool(source['state_weak_narrow'][i])
            active = bool(calm and not weak and source['extension_available'][i, j] and
                source['forecast_available'][i, j] and source['eligible'][i, j] and
                math.isfinite(float(source['rank_log_excess'][i, j])))
            # The original state policy subtracts .025 calm and .05 weak from
            # support; scope67 subtracts .01 only on its inherited active mask.
            # The later local-linear term is a rank difference, not support.
            weight = finite((1. - cw) - (.025 * calm + .05 * weak + .01 * active))
            require(weight > 0., 'Nonpositive effective auxiliary weight')
            values['support'][i, j], values['weight'][i, j] = support, weight
            active_count += int(active)
        q, control, distinct = row_targets(values['support'][i])
        values['candidate'][i], values['control'][i] = q, control
        constant, two_levels = distinct == 1, distinct == 2
        for j in range(width):
            for key in ('candidate', 'control'):
                delta = 0. if i == 0 or constant else finite(float(values['weight'][i, j]) *
                    finite(float(values[key][i, j]) - float(values['support'][i, j])))
                require(abs(delta) <= .5, 'Auxiliary delta exceeds declared bound')
                values[key + '_delta'][i, j] = delta
        rows.append(dict(query_index=i, support_min=float(min(values['support'][i])),
            support_max=float(max(values['support'][i])), candidate_min=min(q), candidate_max=max(q),
            control_min=min(control), control_max=max(control), distinct_levels=distinct,
            constant=constant, two_levels=two_levels, active_forecast_count=active_count,
            max_absolute_candidate_delta=float(max(abs(x) for x in values['candidate_delta'][i])),
            max_absolute_control_delta=float(max(abs(x) for x in values['control_delta'][i])),
            max_absolute_candidate_control_difference=float(max(abs(x-y) for x, y in zip(q, control)))))
    audit = dict(events=n, symbols_per_event=width, checked_cells=n * width, rank_rows=n,
        constant_rows=sum(r['constant'] for r in rows), two_level_rows=sum(r['two_levels'] for r in rows),
        active_forecast_cells=sum(r['active_forecast_count'] for r in rows), new_model_fits=0,
        predictive_model_fits=0, ranking_eligibility_filtered=False, new_availability_filter=False,
        portfolio_returns_calculated=False, event_history=rows)
    return values, audit


def values_check(values, expected):
    require(isinstance(values, dict) and set(values) == set(VALUE_KEYS), 'Six exact feature keys required')
    for key in VALUE_KEYS:
        actual = np.asarray(values[key])
        require(actual.dtype == np.dtype('float64') and actual.shape == expected[key].shape and
                np.array_equal(actual, expected[key]), 'Independent feature differs: ' + key)


def verify_features(arrays, values, audit, limit=None):
    expected, wanted_audit = independent_features(arrays, limit)
    values_check(values, expected)
    record_check(audit, wanted_audit)
    return dict(**{key: True for key in FEATURE_FLAGS},
        **{key: wanted_audit[key] for key in COUNT_FIELDS}, predictive_model_fits=0,
        new_model_fits=0, computes_returns=False, numerical_tolerance=0., support_bound_tolerance=1e-12)


def verify_deltas(values, arrays, deltas, limit=None):
    expected, _ = independent_features(arrays, limit)
    values_check(values, expected)
    require(isinstance(deltas, dict) and set(deltas) == set(DELTA_KEYS), 'Delta arm keys differ')
    changed, maxima = {}, {}
    for arm, key in DELTA_KEYS.items():
        raw = np.asarray(deltas[arm])
        require(raw.dtype == np.dtype('float64') and raw.shape == expected[key].shape and
                np.array_equal(raw, expected[key]), 'Independent full auxiliary delta differs: ' + arm)
        changed[arm] = int(np.count_nonzero(raw))
        maxima[arm] = float(np.max(np.abs(raw))) if raw.size else 0.
    return dict(**{key: True for key in DELTA_FLAGS}, changed_cells=changed,
        max_absolute_delta=.5, observed_max_absolute_delta=max(maxima.values()),
        per_arm_max_absolute_delta=maxima)
