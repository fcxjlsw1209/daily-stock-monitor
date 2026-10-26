"""Source-free all-auxiliary funding bridge to the unchanged episode selector.

Callers establish source admission and incumbent parity before using an active
row. This module loads code only, never actual prices, caches, orders or returns.
An inactive row does not consume target values. A constant active target still
funds existing auxiliaries and is not a promise of incumbent neutrality.
"""
from copy import deepcopy
import hashlib
import importlib.util
import math
from numbers import Real
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EPISODE_PATH = ROOT / 'outputs/monthly_daily_episodes/episodes.py'
EPISODE_SHA256 = '8ae91e55e0d662cb6dee9cba6d1422aa3713b779976de69b4560194d311ae16a'
BUDGET = .02
SUPPORT_VALIDATION_ATOL = 1e-12  # Inherited input check, never a tie tolerance.
COUNT_FIELDS = ('events', 'symbols_per_event', 'active_events', 'inactive_events',
                'consumed_target_cells', 'nonzero_correction_cells',
                'boundary_cells', 'original_episode_calls',
                'new_model_fits', 'predictive_model_fits')
EVENT_KEYS = ('query_index', 'calm', 'weak', 'active', 'target_consumed',
              'core_weight', 'funding_budget', 'funding_coefficient', 'alpha',
              'original_episode_budget', 'requested_episode_budget',
              'effective_episode_budget', 'target', 'correction',
              'original_support', 'boundary_indices', 'chosen_indices')


def _load_episode():
    if hashlib.sha256(EPISODE_PATH.read_bytes()).hexdigest() != EPISODE_SHA256:
        raise ValueError('Frozen episode hash changed')
    spec = importlib.util.spec_from_file_location('reference_gain250_frozen_episode', EPISODE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ep = _load_episode()


def dependency_files():
    """Direct frozen dependency; formal host binds its transitive source closure."""
    return (EPISODE_PATH,)


def _require(condition, label):
    if not condition:
        raise ValueError(label)


def _number(value, label):
    _require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_)),
             label + ' must be real nonboolean')
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(label + ' must convert to binary64') from exc
    _require(math.isfinite(result), label + ' must be finite')
    return result


def _matrix(value, shape, label):
    array = np.asarray(value, dtype=object)
    _require(array.shape == shape, label + ' shape')
    out = np.empty(shape, dtype=np.float64)
    for index in np.ndindex(shape):
        out[index] = _number(array[index], label)
    return out


def _boolean(value, shape, label):
    array = np.asarray(value)
    _require(array.shape == shape and array.dtype.kind == 'b', label + ' boolean shape')
    return array.copy()


def _input_snapshot(value):
    # Do not infer a numeric dtype here: nested inactive target containers may
    # contain objects whose float conversion must never be invoked.
    if isinstance(value, np.ndarray):
        return ('array', value.shape, value.dtype.str, value.tobytes())
    if isinstance(value, (list, tuple)):
        return (type(value), tuple(_input_snapshot(item) for item in value))
    return (type(value), id(value))


def _snapshot(values, symbols):
    return (tuple(_input_snapshot(v) for v in values), deepcopy(symbols))


def score_and_choose(pre_score, core, core_weight, baseline, calm, weak,
                     eligible, symbols, effects, target, active):
    """Return (choice_indices, final_scores, modified_pre_scores, plain_audit).

    P/C/B/effects/eligible/target have shape(n,width); calm/weak/active are bool(n,).
    Original cw accepts(n,), (n,1), or(n,width) and is copied to(n,width).
    No target value on an inactive row is read or converted. The caller may use
    NaN or another placeholder there. Width is retained for the empty prefix.
    """
    inputs = (pre_score, core, core_weight, baseline, calm, weak,
              eligible, effects, target, active)
    before = _snapshot(inputs, symbols)
    shape = np.shape(pre_score)
    _require(len(shape) == 2 and shape[1] > 0, 'Pre-score two-dimensional nonempty width')
    n, width = shape
    p = _matrix(pre_score, shape, 'Pre-score')
    c = _matrix(core, shape, 'Core')
    base = _matrix(baseline, shape, 'Baseline')
    eff = _matrix(effects, shape, 'Effects')
    good = _boolean(eligible, shape, 'Eligibility')
    calm_rows = _boolean(calm, (n,), 'Calm')
    weak_rows = _boolean(weak, (n,), 'Weak')
    on = _boolean(active, (n,), 'Active')
    _require(isinstance(symbols, (list, tuple)) and len(symbols) == n, 'Symbols row axis')
    names = []
    for row in symbols:
        _require(isinstance(row, (list, tuple)) and len(row) == width,
                 'Symbols width')
        _require(all(type(s) is str and s and s.strip() == s for s in row),
                 'Symbols nonempty canonical strings')
        _require(len(set(row)) == width, 'Duplicate symbols')
        names.append(list(row))
    raw_weight = np.asarray(core_weight)
    _require(raw_weight.shape in ((n,), (n, 1), shape), 'Core weight shape')
    weight = _matrix(raw_weight, raw_weight.shape, 'Core weight')
    if weight.ndim == 1:
        weight = weight[:, None]
    cw = np.broadcast_to(weight, shape).copy()
    expected = np.broadcast_to(np.where(calm_rows, .50, .52)[:, None], shape)
    _require(np.array_equal(cw, expected), 'Original common row .50/.52 core weights')
    qraw = np.asarray(target, dtype=object)
    _require(qraw.shape == shape, 'Target shape')
    q = np.full(shape, np.nan, dtype=np.float64)
    for i in np.flatnonzero(on):
        for j in range(width):
            value = _number(qraw[i, j], 'Active target')
            _require(-.5 <= value <= .5, 'Active target bounds')
            q[i, j] = value
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            support = (base - c) / (1. - cw)
            raw_core = c / cw
            _require(np.isfinite(support).all() and
                     (np.abs(support) <= .5 + SUPPORT_VALIDATION_ATOL).all(),
                     'Original support bounds')
            _require(np.isfinite(raw_core).all() and
                     (np.abs(raw_core) <= .5 + SUPPORT_VALIDATION_ATOL).all(),
                     'Original weighted core bounds')
            old_budget = .025 * calm_rows + .05 * weak_rows
            cap = np.maximum(0., .10 - old_budget)
            e0 = np.minimum(.05, cap)
            b = np.where(on, BUDGET, 0.)
            coefficient = np.zeros(n, dtype=np.float64)
            coefficient[on] = BUDGET / (1. - cw[on, 0])
            alpha = 1. - coefficient
            correction = np.zeros(shape, dtype=np.float64)
            # Compute only active rows: inactive targets are genuinely unused.
            correction[on] = BUDGET * q[on] - coefficient[on, None] * (p[on] - c[on])
            modified = p.copy()
            nonzero = correction != 0.
            modified[nonzero] += correction[nonzero]
            requested = np.full(n, .05, dtype=np.float64)
            requested[on] = alpha[on] * e0[on]
            effective = np.minimum(requested, cap)
            if not all(np.isfinite(x).all() for x in
                       (coefficient, alpha, correction, modified, requested, effective)):
                raise ArithmeticError('Nonfinite funding arithmetic')
            original = dict(baseline=base.copy(), core=c.copy(), core_weight=cw.copy(),
                            state_calm10=calm_rows.copy(), state_weak_narrow=weak_rows.copy())
            policy = dict(budget=requested.copy(), ceiling=5, gap=.02)
            # Exactly one call, including n=0. Preserve returned choices and order.
            choices, final = ep.select_boundary(modified.copy(), good.copy(),
                                                deepcopy(names), eff.copy(), original, policy)
    except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
        raise ArithmeticError('Nonfinite funding or episode arithmetic') from exc
    _require(isinstance(choices, list) and len(choices) == n, 'Episode choices axis')
    final = np.asarray(final)
    _require(final.shape == shape, 'Episode score shape')
    if not np.isfinite(final).all():
        raise ArithmeticError('Nonfinite final scores')
    events = []
    boundary_count = 0
    for i in range(n):
        rank = sorted(np.flatnonzero(good[i]), key=lambda j: (-modified[i, j], names[i][j]))
        hook = [] if len(rank) < 3 else [int(j) for j in rank[2:5]
                                       if modified[i, rank[2]] - modified[i, j] <= .02]
        boundary_count += len(hook)
        events.append(dict(query_index=i, calm=bool(calm_rows[i]), weak=bool(weak_rows[i]),
            active=bool(on[i]), target_consumed=bool(on[i]), core_weight=float(cw[i, 0]),
            funding_budget=float(b[i]), funding_coefficient=float(coefficient[i]), alpha=float(alpha[i]),
            original_episode_budget=float(e0[i]), requested_episode_budget=float(requested[i]),
            effective_episode_budget=float(effective[i]), target=q[i].tolist() if on[i] else None,
            correction=correction[i].tolist(), original_support=support[i].tolist(),
            boundary_indices=hook, chosen_indices=[int(j) for j in choices[i]]))
    counts = dict(events=n, symbols_per_event=width, active_events=int(on.sum()),
                  inactive_events=int((~on).sum()), consumed_target_cells=int(on.sum()) * width,
                  nonzero_correction_cells=int(nonzero.sum()), boundary_cells=boundary_count,
                  original_episode_calls=1, new_model_fits=0, predictive_model_fits=0)
    audit = dict(schema='monthly_reference_gain250_funding_adapter_v1', counts=counts,
                 source_free=True, actual_data_loaded=False, target_inactive_unconsumed=True,
                 constant_target_is_not_neutral=True, original_inputs_unchanged=True, events=events)
    _require(tuple(counts) == COUNT_FIELDS and all(tuple(e) == EVENT_KEYS for e in events), 'Audit schema')
    _require(before == _snapshot(inputs, symbols), 'Caller input mutation')
    return choices, final.copy(), modified, audit
