"""Full PRE-episode support remapping of the frozen Local67 cached aggregate.

The caller adds these deltas to its original pre-score, then applies exactly one
original episode with ORIGINAL support. This is not a post-episode remapping.
Pure cache transforms only: no source I/O, models, eligibility or portfolio work.
"""
import numbers

import numpy as np
import pandas as pd

REQUIRED = {'baseline', 'core', 'core_weight', 'state_calm10', 'state_weak_narrow',
            'extension_available', 'forecast_available', 'eligible', 'rank_log_excess'}
VALUE_KEYS = ('support', 'weight', 'candidate', 'control', 'candidate_delta', 'control_delta')
BOUND_TOL = 1e-12


def _prefix(value, shape, n, name, boolean=False, finite=True):
    source = np.asarray(value)
    if source.shape != shape:
        raise ValueError(name + ' shape differs from original event axes')
    # Slice before inspecting values or coercing numeric/object storage. Future
    # poison must not be inspected just because it shares a source array.
    raw = source[:n]
    if boolean:
        if any(not isinstance(x, (bool, np.bool_)) for x in raw.flat):
            raise ValueError(name + ' requires exact boolean values')
        return raw.astype(bool, copy=True)
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real) for x in raw.flat):
        raise ValueError(name + ' requires real nonboolean numeric values')
    try:
        result = raw.astype(np.float64, copy=True)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(name + ' cannot be represented as float64') from exc
    if finite and not np.isfinite(result).all():
        raise ValueError(name + ' requires finite values')
    return result


def _sources(arrays, limit):
    if not isinstance(arrays, dict) or not REQUIRED <= set(arrays):
        raise ValueError('Original cached support, weights, states and masks required')
    shape = np.asarray(arrays['baseline']).shape
    if len(shape) != 2 or shape[1] < 1:
        raise ValueError('Original baseline requires an event matrix of positive width')
    rows, width = shape
    if limit is not None and (type(limit) is not int or not 0 <= limit <= rows):
        raise ValueError('limit must be an integer event prefix')
    n = rows if limit is None else limit
    baseline = _prefix(arrays['baseline'], shape, n, 'baseline')
    core = _prefix(arrays['core'], shape, n, 'core')
    calm = _prefix(arrays['state_calm10'], (rows,), n, 'state_calm10', boolean=True)
    weak = _prefix(arrays['state_weak_narrow'], (rows,), n, 'state_weak_narrow', boolean=True)
    raw_weight = np.asarray(arrays['core_weight'])
    if raw_weight.shape == (rows,):
        raw_weight = raw_weight[:, None]
    if raw_weight.shape not in ((rows, 1), shape):
        raise ValueError('core_weight shape must be n, n-by-1 or n-by-width')
    cw = np.broadcast_to(_prefix(raw_weight, raw_weight.shape, n, 'core_weight'), (n, width)).copy()
    expected = np.broadcast_to(np.where(calm, .50, .52)[:, None], cw.shape)
    if not np.array_equal(cw, expected):
        raise ValueError('core_weight disagrees with original calm state')
    extension = _prefix(arrays['extension_available'], shape, n, 'extension_available', boolean=True)
    forecast = _prefix(arrays['forecast_available'], shape, n, 'forecast_available', boolean=True)
    eligible = _prefix(arrays['eligible'], shape, n, 'eligible', boolean=True)
    logrank = _prefix(arrays['rank_log_excess'], shape, n, 'rank_log_excess', finite=False)
    active = calm[:, None] & ~weak[:, None] & extension & forecast & eligible & np.isfinite(logrank)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        support = (baseline - core) / (1. - cw)
        weight = (1. - cw) - (.025 * calm[:, None] + .05 * weak[:, None] + .01 * active)
    if not np.isfinite(support).all() or (np.abs(support) > .5 + BOUND_TOL).any():
        raise ValueError('Original auxiliary support exceeds finite bounded domain')
    if not np.isfinite(weight).all() or (weight <= 0.).any():
        raise ValueError('Effective original support weight must be finite positive')
    return support, weight, active


def _targets(support):
    """Exact full-row tie ranks and affine control with identical endpoints."""
    width = len(support)
    low, high = float(support.min()), float(support.max())
    levels = int(len(np.unique(support)))
    if levels == 1:
        return np.zeros(width), np.zeros(width), levels
    candidate = (pd.Series(support).rank(method='average').to_numpy() - 1.) / (width - 1.) - .5
    qlow, qhigh = float(candidate.min()), float(candidate.max())
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        control = qlow + (support - low) * (qhigh - qlow) / (high - low)
    control[support == low] = qlow
    control[support == high] = qhigh
    if levels == 2:
        control = candidate.copy()
    if not (np.isfinite(candidate).all() and np.isfinite(control).all()):
        raise FloatingPointError('Nonfinite auxiliary rank transformation')
    if (np.abs(candidate) > .5 + BOUND_TOL).any() or (np.abs(control) > .5 + BOUND_TOL).any():
        raise FloatingPointError('Auxiliary transformation exceeds bounded range')
    return candidate, control, levels


def build_features(arrays, limit=None):
    """Return six float64 matrices and recorded statistics for a causal prefix."""
    support, weight, active = _sources(arrays, limit)
    n, width = support.shape
    values = {key: np.zeros((n, width), dtype=np.float64) for key in VALUE_KEYS}
    values['support'] = support
    values['weight'] = weight
    history = []
    constant_rows = two_level_rows = 0
    for i, row in enumerate(support):
        candidate, control, levels = _targets(row)
        values['candidate'][i] = candidate
        values['control'][i] = control
        constant, two = levels == 1, levels == 2
        constant_rows += int(constant)
        two_level_rows += int(two)
        # Constant rows are whole-row no-ops even when their common S is nonzero.
        if i and not constant:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                values['candidate_delta'][i] = weight[i] * (candidate - row)
                values['control_delta'][i] = weight[i] * (control - row)
        history.append(dict(query_index=i, support_min=float(row.min()), support_max=float(row.max()),
            candidate_min=float(candidate.min()), candidate_max=float(candidate.max()),
            control_min=float(control.min()), control_max=float(control.max()), distinct_levels=levels,
            constant=constant, two_levels=two, active_forecast_count=int(active[i].sum()),
            max_absolute_candidate_delta=float(np.abs(values['candidate_delta'][i]).max()),
            max_absolute_control_delta=float(np.abs(values['control_delta'][i]).max()),
            max_absolute_candidate_control_difference=float(np.abs(candidate - control).max())))
    if any(not np.isfinite(value).all() for value in values.values()):
        raise FloatingPointError('Nonfinite cached auxiliary transform')
    return values, dict(events=n, symbols_per_event=width, checked_cells=n*width, rank_rows=n,
        constant_rows=constant_rows, two_level_rows=two_level_rows, active_forecast_cells=int(active.sum()),
        new_model_fits=0, predictive_model_fits=0, ranking_eligibility_filtered=False,
        new_availability_filter=False, portfolio_returns_calculated=False, event_history=history)
