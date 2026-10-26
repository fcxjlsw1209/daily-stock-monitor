"""Stock/ETF grouped PRE-episode support remapping of Local67 cached aggregate.

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
    """Exact within-group tie ranks and affine control with equal endpoints."""
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


def _groups(meta, rows, width, n):
    if not isinstance(meta, dict) or not {'symbols', 'etfs'} <= set(meta):
        raise ValueError('Explicit symbol rows and ETF identity list required')
    etfs = meta['etfs']
    if (not isinstance(etfs, list) or
            any(not isinstance(x, str) or not x for x in etfs) or
            len(set(etfs)) != len(etfs)):
        raise ValueError('etfs must be an explicit unique nonempty-string list')
    symbols = meta['symbols']
    if not isinstance(symbols, list) or len(symbols) != rows:
        raise ValueError('symbols must retain the original full event axis')
    selected = []
    for row in symbols[:n]:
        if (not isinstance(row, list) or len(row) != width or
                any(not isinstance(x, str) or not x for x in row) or
                len(set(row)) != width):
            raise ValueError('Each consumed symbols row must contain unique explicit identities')
        selected.append(row.copy())
    return selected, set(etfs)


def _maximum(values):
    return float(np.abs(values).max()) if len(values) else 0.


def build_features(arrays, meta, limit=None):
    """Return six float64 matrices, ranking all candidates within asset type.

    Trading eligibility is used only in the inherited effective funding weight.
    Group membership comes solely from the explicit global metadata ETF list.
    Empty, singleton and constant groups are neutral. No new availability mask
    or source/model input is constructed. The caller preserves original episode
    support and applies that original selector once after adding nonzero deltas.
    """
    support, weight, active = _sources(arrays, limit)
    n, width = support.shape
    symbols, etfs = _groups(meta, np.asarray(arrays['baseline']).shape[0], width, n)
    values = {key: np.zeros((n, width), dtype=np.float64) for key in VALUE_KEYS}
    values['support'], values['weight'] = support, weight
    history = []
    ranked = constants = two_levels = empty = stocks = funds = 0
    for i, (row, names) in enumerate(zip(support, symbols)):
        group_history = []
        for asset_type in ('stock', 'etf'):
            ids = [j for j, name in enumerate(names) if (name in etfs) == (asset_type == 'etf')]
            subset = row[ids]
            size = len(ids)
            stocks += size if asset_type == 'stock' else 0
            funds += size if asset_type == 'etf' else 0
            if size:
                candidate, control, levels = _targets(subset)
                constant, two = levels == 1, levels == 2
                ranked += 1
                constants += int(constant)
                two_levels += int(two)
                values['candidate'][i, ids] = candidate
                values['control'][i, ids] = control
                # A nonzero constant S must not be subtracted just to center it.
                if i and not constant:
                    with np.errstate(over='raise', invalid='raise', divide='raise'):
                        values['candidate_delta'][i, ids] = weight[i, ids] * (candidate - subset)
                        values['control_delta'][i, ids] = weight[i, ids] * (control - subset)
                bounds = dict(support_min=float(subset.min()), support_max=float(subset.max()),
                    candidate_min=float(candidate.min()), candidate_max=float(candidate.max()),
                    control_min=float(control.min()), control_max=float(control.max()))
            else:
                empty += 1
                candidate = control = np.zeros(0, dtype=np.float64)
                levels, constant, two = 0, False, False
                bounds = {key: None for key in ('support_min', 'support_max', 'candidate_min',
                                               'candidate_max', 'control_min', 'control_max')}
            group_history.append(dict(asset_type=asset_type, indices=ids,
                symbols=[names[j] for j in ids], size=size, **bounds,
                distinct_levels=levels, constant=constant, two_levels=two,
                active_forecast_count=int(active[i, ids].sum()),
                max_absolute_candidate_delta=_maximum(values['candidate_delta'][i, ids]),
                max_absolute_control_delta=_maximum(values['control_delta'][i, ids]),
                max_absolute_candidate_control_difference=_maximum(candidate-control)))
        history.append(dict(query_index=i, symbols=names, groups=group_history,
            active_forecast_count=int(active[i].sum()),
            max_absolute_candidate_delta=_maximum(values['candidate_delta'][i]),
            max_absolute_control_delta=_maximum(values['control_delta'][i]),
            max_absolute_candidate_control_difference=_maximum(values['candidate'][i]-values['control'][i])))
    if any(not np.isfinite(value).all() for value in values.values()):
        raise FloatingPointError('Nonfinite grouped auxiliary transform')
    return values, dict(events=n, symbols_per_event=width, checked_cells=n*width,
        rank_rows=ranked, grouped_rank_rows=ranked, constant_rows=constants,
        two_level_rows=two_levels, empty_groups=empty, stock_cells=stocks, etf_cells=funds,
        active_forecast_cells=int(active.sum()), new_model_fits=0, predictive_model_fits=0,
        ranking_eligibility_filtered=False, new_availability_filter=False,
        portfolio_returns_calculated=False, event_history=history)


build = build_features


